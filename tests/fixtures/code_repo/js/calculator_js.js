// A tiny JavaScript calculator used to test how KíLín reads code.

const js_function_B = (x) => 2 * x;

/**
 * Double a, then add b.
 */
function js_function_A(a, b) {
  return js_function_B(a) + b;
}

class JsBaseClass {
  constructor(value) {
    this.value = value;
  }
}

// Adds b to the stored value, doubled.
class JsChildClass extends JsBaseClass {
  js_method_C(b) {
    return Math.round(js_function_A(this.value, b));
  }
}

/**
 * Apply js_method_C to every number in a list.
 */
function js_function_D(values) {
  const calculator = new JsChildClass(1);
  return values.map((b) => calculator.js_method_C(b));
}
