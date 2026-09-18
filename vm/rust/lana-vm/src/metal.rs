//! Explicit GPU matmul (LIP-004 section 5), mirroring `vm/c/metal.m`. A single
//! float32 sgemm on the system Metal device. The shader is embedded from
//! `vm/metal/matmul.metal` (the same source the C VM compiles), so both VMs run
//! the same kernel and produce identical results on a device.

#[cfg(target_os = "macos")]
mod imp {
    use std::os::raw::c_void;
    use std::sync::OnceLock;

    use metal::*;

    pub struct ResidentBuffer { buffer: Buffer, len: usize }

    impl std::fmt::Debug for ResidentBuffer {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("ResidentBuffer").field("len", &self.len).finish()
        }
    }

    impl ResidentBuffer {
        pub fn new(bytes: &[u8]) -> Option<Self> {
            let (device, _) = pipeline()?;
            let buffer = if bytes.is_empty() { device.new_buffer(0, MTLResourceOptions::StorageModeShared) }
                else { device.new_buffer_with_data(bytes.as_ptr() as *const c_void, bytes.len() as u64, MTLResourceOptions::StorageModeShared) };
            Some(Self { buffer, len: bytes.len() })
        }
        pub(crate) fn copy_into(&self, bytes: &mut [u8]) {
            assert_eq!(bytes.len(), self.len, "resident buffer length changed");
            if self.len > 0 { unsafe { std::ptr::copy_nonoverlapping(self.buffer.contents() as *const u8, bytes.as_mut_ptr(), self.len); } }
        }
    }

    const SHADER: &str = include_str!("../../../metal/matmul.metal");

    /// The device and compiled pipeline are cached across calls (the VM is
    /// single-threaded). `None` means no Metal device is available.
    fn pipeline() -> Option<&'static (Device, ComputePipelineState)> {
        static PIPELINE: OnceLock<Option<(Device, ComputePipelineState)>> = OnceLock::new();
        PIPELINE.get_or_init(|| {
            let device = Device::system_default()?;
            let library = device
                .new_library_with_source(SHADER, &CompileOptions::new())
                .ok()?;
            let function = library.get_function("sgemm", None).ok()?;
            let pipeline = device
                .new_compute_pipeline_state_with_function(&function)
                .ok()?;
            Some((device, pipeline))
        })
        .as_ref()
    }

    pub fn sgemm(m: usize, k: usize, n: usize, a: *const f32, b: *const f32, c: *mut f32) -> bool {
        if m == 0 || n == 0 {
            return true; // nothing to write
        }
        if m > u32::MAX as usize || k > u32::MAX as usize || n > u32::MAX as usize {
            return false;
        }
        let Some((device, pipeline)) = pipeline() else {
            return false;
        };
        let queue = device.new_command_queue();
        let a_bytes = (m * k * std::mem::size_of::<f32>()) as u64;
        let b_bytes = (k * n * std::mem::size_of::<f32>()) as u64;
        let c_bytes = (m * n * std::mem::size_of::<f32>()) as u64;
        let buf_a = device.new_buffer_with_data(a as *const c_void, a_bytes, MTLResourceOptions::StorageModeShared);
        let buf_b = device.new_buffer_with_data(b as *const c_void, b_bytes, MTLResourceOptions::StorageModeShared);
        let buf_c = device.new_buffer(c_bytes, MTLResourceOptions::StorageModeShared);
        let dims: [u32; 3] = [m as u32, k as u32, n as u32];
        let buf_dims = device.new_buffer_with_data(
            dims.as_ptr() as *const c_void,
            std::mem::size_of_val(&dims) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&buf_a), 0);
        encoder.set_buffer(1, Some(&buf_b), 0);
        encoder.set_buffer(2, Some(&buf_c), 0);
        encoder.set_buffer(3, Some(&buf_dims), 0);
        encoder.set_buffer(4, Some(&buf_dims), 4);
        encoder.set_buffer(5, Some(&buf_dims), 8);

        let threads = MTLSize::new(16, 16, 1);
        let groups = MTLSize::new(((m + 15) / 16) as u64, ((n + 15) / 16) as u64, 1);
        encoder.dispatch_thread_groups(groups, threads);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        if command_buffer.status() == MTLCommandBufferStatus::Error {
            return false;
        }
        unsafe {
            std::ptr::copy_nonoverlapping(buf_c.contents() as *const f32, c, m * n);
        }
        true
    }

    pub fn available() -> bool { pipeline().is_some() }
}

#[cfg(not(target_os = "macos"))]
mod imp {
    #[derive(Debug)]
    pub struct ResidentBuffer;
    impl ResidentBuffer {
        pub fn new(_: &[u8]) -> Option<Self> { None }
        pub(crate) fn copy_into(&self, _: &mut [u8]) { unreachable!("Metal buffer on unsupported platform") }
    }
    pub fn sgemm(_: usize, _: usize, _: usize, _: *const f32, _: *const f32, _: *mut f32) -> bool {
        false
    }
    pub fn available() -> bool { false }
}

pub use imp::ResidentBuffer;

/// One row-major float32 general matrix multiply: `c = a . b` where `a` is
/// `m x k`, `b` is `k x n`, and `c` is `m x n`. Returns `false` when no Metal
/// device is available or a Metal call fails; the caller maps that to
/// `LanaError::UnsupportedOperation`.
pub fn metal_sgemm(m: usize, k: usize, n: usize, a: *const f32, b: *const f32, c: *mut f32) -> bool {
    imp::sgemm(m, k, n, a, b, c)
}

pub fn metal_available() -> bool { imp::available() }
