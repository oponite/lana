//! Bounded, byte-preserving local text and Markdown extraction.

use std::io::Read;
use std::path::Path;
use lana_bytecode::LanaError;

const MAX_INPUT: usize = 16 * 1024 * 1024;
const MAX_CHUNK: usize = 4096;
const MAX_CHUNKS: usize = 100_000;

pub struct Chunk {
    pub text: String,
    pub start_line: usize,
    pub end_line: usize,
    pub start_byte: usize,
    pub end_byte: usize,
    pub heading_path: Vec<String>,
}

pub struct Document {
    pub format: String,
    pub sha256: String,
    pub chunks: Vec<Chunk>,
    pub status: &'static str,
}

fn breaks_before(bytes: &[u8], end: usize) -> usize {
    let mut count = 0;
    let mut index = 0;
    while index < end {
        match bytes[index] {
            b'\r' if bytes.get(index + 1) == Some(&b'\n') => {
                if index + 2 > end { break; }
                count += 1;
                index += 2;
            }
            b'\r' | b'\n' => { count += 1; index += 1; }
            _ => index += 1,
        }
    }
    count
}

fn emit(source: &str, start: usize, end: usize, start_line: usize, heading_path: &[String], chunks: &mut Vec<Chunk>) -> Result<(), LanaError> {
    let bytes = source.as_bytes();
    let mut cursor = start;
    let mut line = start_line;
    while cursor < end {
        if chunks.len() == MAX_CHUNKS { return Err(LanaError::Limit); }
        let mut split = (cursor + MAX_CHUNK).min(end);
        while !source.is_char_boundary(split) || (split < end && bytes[split - 1] == b'\r' && bytes[split] == b'\n') {
            split -= 1;
        }
        if split == cursor { return Err(LanaError::Limit); }
        let segment = &bytes[cursor..split];
        let end_line = line + breaks_before(segment, segment.len() - 1);
        chunks.push(Chunk {
            text: source[cursor..split].to_string(), start_line: line, end_line,
            start_byte: cursor, end_byte: split, heading_path: heading_path.to_vec(),
        });
        line += breaks_before(segment, segment.len());
        cursor = split;
    }
    Ok(())
}

fn fence_open(line: &str) -> Option<(u8, usize)> {
    let spaces = line.bytes().take_while(|byte| *byte == b' ').count();
    if spaces > 3 { return None; }
    let line = &line.as_bytes()[spaces..];
    let marker = *line.first()?;
    if marker != b'`' && marker != b'~' { return None; }
    let width = line.iter().take_while(|byte| **byte == marker).count();
    (width >= 3).then_some((marker, width))
}

fn fence_close(line: &str, marker: u8, width: usize) -> bool {
    let spaces = line.bytes().take_while(|byte| *byte == b' ').count();
    if spaces > 3 { return false; }
    let line = &line.as_bytes()[spaces..];
    let count = line.iter().take_while(|byte| **byte == marker).count();
    count >= width && line[count..].iter().all(|byte| *byte == b' ' || *byte == b'\t')
}

pub fn extract(path: &Path, format: &str) -> Result<Document, LanaError> {
    if format != "text" && format != "markdown" { return Err(LanaError::InvalidParameters); }
    let mut file = std::fs::File::open(path).map_err(|_| LanaError::Io)?;
    let mut bytes = Vec::new();
    file.by_ref().take((MAX_INPUT + 1) as u64).read_to_end(&mut bytes).map_err(|_| LanaError::Io)?;
    if bytes.len() > MAX_INPUT { return Err(LanaError::Limit); }
    if bytes.contains(&0) { return Err(LanaError::Format); }
    let source = std::str::from_utf8(&bytes).map_err(|_| LanaError::Format)?;
    let sha256 = crate::sha256::sha256(&bytes).iter().map(|byte| format!("{byte:02x}")).collect();
    let mut chunks = Vec::new();
    let mut headings: Vec<(usize, String)> = Vec::new();
    let mut paragraph: Option<(usize, usize, usize, Vec<String>)> = None;
    let mut fence: Option<(usize, u8, usize, usize, Vec<String>)> = None;
    let mut cursor = 0;
    let mut line_number = 1;
    while cursor < bytes.len() {
        let mut content_end = cursor;
        while content_end < bytes.len() && bytes[content_end] != b'\r' && bytes[content_end] != b'\n' { content_end += 1; }
        let end = if content_end == bytes.len() { content_end }
            else if bytes[content_end] == b'\r' && bytes.get(content_end + 1) == Some(&b'\n') { content_end + 2 }
            else { content_end + 1 };
        let content = &source[cursor..content_end];
        if let Some((start, marker, width, start_line, path)) = fence.as_ref() {
            if fence_close(content, *marker, *width) {
                emit(source, *start, end, *start_line, path, &mut chunks)?;
                fence = None;
            }
        } else if format == "markdown" && fence_open(content).is_some() {
            if let Some((start, last, first_line, path)) = paragraph.take() { emit(source, start, last, first_line, &path, &mut chunks)?; }
            let (marker, width) = fence_open(content).unwrap();
            fence = Some((cursor, marker, width, line_number, headings.iter().map(|(_, text)| text.clone()).collect()));
        } else {
            let heading = if format == "markdown" {
                let width = content.bytes().take_while(|byte| *byte == b'#').count();
                (1..=6).contains(&width).then(|| content.as_bytes().get(width).filter(|byte| **byte == b' ').map(|_| (width, content[width + 1..].trim().to_string()))).flatten()
            } else { None };
            if content.trim().is_empty() || heading.is_some() {
                if let Some((start, last, first_line, path)) = paragraph.take() { emit(source, start, last, first_line, &path, &mut chunks)?; }
                if let Some((level, text)) = heading {
                    while headings.last().is_some_and(|(old, _)| *old >= level) { headings.pop(); }
                    headings.push((level, text));
                    let path = headings.iter().map(|(_, text)| text.clone()).collect::<Vec<_>>();
                    emit(source, cursor, end, line_number, &path, &mut chunks)?;
                }
            } else if let Some((_, last, _, _)) = paragraph.as_mut() {
                *last = end;
            } else {
                paragraph = Some((cursor, end, line_number, headings.iter().map(|(_, text)| text.clone()).collect()));
            }
        }
        cursor = end;
        line_number += 1;
    }
    if let Some((start, _, _, first_line, path)) = fence { emit(source, start, bytes.len(), first_line, &path, &mut chunks)?; }
    if let Some((start, last, first_line, path)) = paragraph { emit(source, start, last, first_line, &path, &mut chunks)?; }
    Ok(Document { format: format.to_string(), sha256, chunks,
        status: if bytes.is_empty() { "empty" } else { "exact_text" } })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn markdown_spans_and_long_unicode_chunks_preserve_source_bytes() {
        let path = std::env::temp_dir().join(format!("lana-document-{}", std::process::id()));
        let source = "# Top\r\nalpha 🐈\r\nbeta\n\n## Child\n```rust\nx\n\n```\n\ntail";
        std::fs::write(&path, source).unwrap();
        let document = extract(&path, "markdown").unwrap();
        assert_eq!(document.status, "exact_text");
        assert_eq!(document.chunks.len(), 5);
        for chunk in &document.chunks {
            assert_eq!(&source[chunk.start_byte..chunk.end_byte], chunk.text);
        }
        assert_eq!((document.chunks[1].start_line, document.chunks[1].end_line), (2, 3));
        assert_eq!(document.chunks[1].heading_path, ["Top"]);
        assert_eq!(document.chunks[3].heading_path, ["Top", "Child"]);
        assert_eq!(document.chunks[3].text, "```rust\nx\n\n```\n");
        let long = "é".repeat(3000);
        std::fs::write(&path, &long).unwrap();
        let document = extract(&path, "text").unwrap();
        assert_eq!(document.chunks.len(), 2);
        assert_eq!(document.chunks[0].end_byte, 4096);
        assert_eq!(document.chunks[1].start_byte, 4096);
        for chunk in &document.chunks { assert_eq!(&long[chunk.start_byte..chunk.end_byte], chunk.text); }
        std::fs::write(&path, "one\r\n\r\ntwo\n").unwrap();
        let plain = extract(&path, "text").unwrap();
        assert_eq!(plain.chunks.len(), 2);
        assert_eq!(plain.chunks[0].text, "one\r\n");
        assert_eq!((plain.chunks[1].start_byte, plain.chunks[1].start_line), (7, 3));
        std::fs::write(&path, []).unwrap();
        let empty = extract(&path, "text").unwrap();
        assert_eq!(empty.status, "empty");
        assert!(empty.chunks.is_empty());
        std::fs::write(&path, [0xff]).unwrap();
        assert!(matches!(extract(&path, "text"), Err(LanaError::Format)));
        std::fs::write(&path, [0]).unwrap();
        assert!(matches!(extract(&path, "text"), Err(LanaError::Format)));
        std::fs::File::create(&path).unwrap().set_len((MAX_INPUT + 1) as u64).unwrap();
        assert!(matches!(extract(&path, "text"), Err(LanaError::Limit)));
        assert!(matches!(extract(&path, "pdf"), Err(LanaError::InvalidParameters)));
        std::fs::remove_file(path).unwrap();
    }
}
