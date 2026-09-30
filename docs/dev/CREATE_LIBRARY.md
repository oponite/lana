# Create a Lana library

1. Create a project:

   ```sh
   lana new my-library
   cd my-library
   ```

2. Create `src/math.lana`:

   ```lana
   fn twice(value) {
       return value * 2;
   }
   ```

   Imported modules cannot execute top-level statements.
   Put reusable behavior inside functions.

3. Replace `src/main.lana` with an example that uses the library:

   ```lana
   import "./math.lana" as math;

   print(math.twice(3));
   ```

4. Create `tests/math_test.lana`:

   ```lana
   import "../src/math.lana" as math;

   assert(math.twice(3) == 6, "twice returns double the value");
   ```

5. Run the example and tests:

   ```sh
   lana run
   lana test
   ```

   The example prints `6`. The tests must pass.

To share the library, follow [SOURCE_PACKAGES.md](SOURCE_PACKAGES.md).