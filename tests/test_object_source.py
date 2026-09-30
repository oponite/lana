"""End-to-end source object model and rejection boundaries."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

cli = str(Path(sys.argv[1]).resolve())
with tempfile.TemporaryDirectory(prefix="lana-object-source-") as directory:
    root = Path(directory)
    def run_source(source, output=None, failure=None, command="run"):
        path = root / "main.lana"
        path.write_text(source)
        arguments = [cli, command, str(path)]
        if command == "compile":
            arguments.extend(["-o", str(root / "compiled.labc")])
        result = subprocess.run(arguments, text=True, capture_output=True, timeout=60)
        if failure is not None:
            assert result.returncode, (source, result)
            assert failure in result.stderr, (source, result.stderr)
        else:
            assert result.returncode == 0, (source, result.stderr)
            if output is not None:
                assert result.stdout.strip() == output, (source, result.stdout)
        return result

    base = '''
interface Readable { public fn read(self) -> number; }
value Reading implements Readable {
    private number: number;
    public static fn create(number: number) -> Self { return Self(number); }
    public fn read(self) -> number { return self.number; }
}
class Counter implements Readable {
    private mutable number: number;
    private fn init(self, number: number) { self.number = number; }
    public static fn create(number: number) -> Self { return new Self(number); }
    public fn read(self) -> number { return self.number; }
    public fn set(self, number: number) -> null { self.number = number; }
}
'''
    run_source(base + '''
fn read(view: Readable) -> number { return view.read(); }
let v = Reading.create(7);
print(read(v));
let c = Counter.create(8);
c.set(9);
let view: Readable = c;
print(read(view));
print(c == view);
''', "7\n9\ntrue")
    copied = '''
class Parent {
    private count: number = 2;
    public fn read(self) -> number { return self.count; }
    public fn via(self) -> number { return self.read(); }
    public static fn create() -> Self { return new Self(); }
}
class Child copies Parent {
    replace private count: number = 3;
    replace public fn read(self) -> number { return self.count + 4; }
}
'''
    run_source(copied + "let c = Child.create(); print(c.via()); print(c == c);", "7\ntrue")
    run_source(copied + "let p: Parent = Child.create();", failure="binding type mismatch", command="compile")
    run_source(copied.replace("replace private count: number = 3;", "replace private count: bool = true;")
               .replace("replace public fn read(self) -> number { return self.count + 4; }", ""),
               failure="expected number", command="compile")
    run_source(base + "let c = Counter.create(1); print(c.number);", failure="private member", command="compile")
    run_source(base + "let c = new Counter(1);", failure="private member", command="compile")
    run_source(base + "let v = Reading(1);", failure="private member", command="compile")
    for source, error in [
        ("value V { x: number; }", "public or private"),
        ("value V { public mutable x: number; }", "cannot be mutable"),
        ("value V { public x: number = 1; }", "cannot have defaults"),
        ("value V<T> {}", "expected {"),
        ("class C { public fn f(self, x) -> null {} }", "expected :"),
        ("class C { public fn f(self) {} }", "expected ->"),
        ("class C { public fn init(self) -> null {} }", "expected {"),
        ("class C { public fn init(self) {} public fn init(self) {} }", "duplicate method"),
        ("class C { public fn f<T>(self) -> null {} }", "expected ("),
        ("interface I { public static fn f() -> null; }", "cannot be static"),
        ("interface I { private fn f(self) -> null; }", "must be public"),
        ("interface I implements J {}", "cannot inherit"),
        ("fn f() { class C {} }", "module top level"),
        ("class C { replace public x: number = 1; }", "copied member"),
        ("class C { public x: number = 1; public fn x(self) -> number { return 1; } }", "names conflict"),
        ("class C { public static fn f(x: STATE) -> null {} public static fn f(x: STATE_DIST) -> null {} }", "cannot overload"),
        ("class C { public fn f(self, x: number) -> null {} public fn f(self, x: string) -> null {} }", "unsupported overload"),
        ("class C { public x: number; } let c = new C();", "requires defaults"),
        ("class C { public x: number = now(); }", "defaults must be pure"),
        ("class C { public fn init(self) { print(1); } }", "init may only"),
        ("class C { public x: number = 1; } let c = new C(); c.x = 2;", "fixed field"),
        ("interface I { public fn f(self) -> null; } class C implements I { public fn f(self) -> null { print(1); } }", "effect promise"),
        ("interface I { public fn f(self) -> null effects(io, io); }", "duplicate interface effect"),
    ]:
        run_source(source, failure=error, command="compile")

    (root / "lib.lana").write_text('''
interface Readable { public fn read(self) -> number; }
class Box implements Readable {
    private value: number = 12;
    public fn read(self) -> number { return self.value; }
    public static fn create() -> Self { return new Self(); }
}
''')
    run_source('import "./lib.lana" as lib; let b = lib.Box.create(); let i: lib.Readable = b; print(i.read());', "12")
    run_source('import "./lib.lana" as lib; class Child copies lib.Box {}', failure="imported class", command="compile")
    run_source('''
value Reading { public state: STATE; public fn read(self) -> STATE { return self.state; } }
class Sensor { public mutable state: STATE; public fn init(self, state: STATE) { self.state = state; } public fn read(self) -> STATE { return self.state; } }
state belief = state(p: 0.5, d_re: 0.3, d_im: 0.4);
let value = Reading(belief);
let sensor = new Sensor(belief);
print(value.read() == sensor.read());
''', "true")
    run_source('''
class Link { public mutable link: Dynamic = null; }
fn echo(value: Link) -> Link { return value; }
let source = new Link(); source.link = source;
let task = fork echo(source);
let copied: Link = join(task);
print(source == copied);
let link: Link = copied.link;
print(copied == link);
''', "false\ntrue")
    run_source('''
value V { public n: number; public fn read(self) -> number { return self.n; } }
let one = V(1); let two = V(2);
let options: Information<V> = possibility([one, two]);
let result = options.read();
print(inspect_information(result).form);
''', "possibility")
    run_source('''
value V {
    public fn pick(self, x: Information<number>) -> number { return resolve(x); }
    public fn pick(self, x: STATE) -> number { return 2; }
    public fn pick(self, x: STATE_DIST) -> number { return 3; }
}
let v = V(); let x: Information<number> = information(7); print(v.pick(x));
''', "7")
    run_source('''
class C { public x: number; public fn init(self) {} }
let c = new C();
''', failure="LANA_ERR_TYPE")
    run_source("value V { public n: number; public fn same(self, other: Self) -> bool { return self == other; } } print(V(1).same(V(1)));", "true")
    run_source('''
class Parent {
    public fn pick(self, x: STATE) -> number { return 1; }
    public fn pick(self, x: Information<number>) -> number { return 2; }
}
class Child copies Parent { replace public fn pick(self, x: STATE) -> number { return 3; } }
state belief = state(p: 0.5, d: 0.3);
let child = new Child(); print(child.pick(belief)); print(child.pick(information(7)));
''', "3\n2")
    run_source('''
class Inner { public n: number; public fn init(self, n: number) { self.n = n; } }
class Outer { public inner: Inner; public fn init(self, n: number) { self.inner = new Inner(n); } }
let outer = new Outer(11); print(outer.inner.n);
''', "11")
    run_source("class C { public mutable link: Self; public fn init(self) { self.link = self; } } let a = new C(); let b = new C(); a.link = b; print(a.link == b);", "true")
    run_source("value V {} print(V());", "value<file/main.lana/V>")
    run_source('let value = 7; let class = 8; let interface = 9; print(value + class + interface);', "24")
print("OBJECT_SOURCE_PASS")
