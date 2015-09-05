package com.simiacryptus.mindseye.math;

/***
 * Represents a real number using a double-precision floating point value
 * in the logrithmic scale, providing extremely high dynamic range in
 * arithmetic operations
 *
 * @author Andrew Charneski
 */
@SuppressWarnings("serial")
public class LogNumber extends Number implements Comparable<LogNumber> {
  public static final int ADDITION_PRECISION = 5;
  public static final LogNumber ONE = new LogNumber((byte) 1, 0);
  
  public static final LogNumber ZERO = new LogNumber((byte) 0, 0);
  
  public static LogNumber log(final double v) {
    if (0. == Math.abs(v)) return new LogNumber((byte) 0, 0);
    final LogNumber logNumber = new LogNumber((byte) (v < 0. ? -1 : 1), Math.log(Math.abs(v)));
    // assert Math.abs(Math.log(logNumber.doubleValue() / v)) < 0.001;
    return logNumber;
  }
  
  public final double logValue;
  
  public final byte type;

  private LogNumber(final byte type, final double logValue) {
    super();
    this.type = type;
    this.logValue = logValue;
  }

  public LogNumber abs() {
    return new LogNumber((byte) 1, this.logValue);
  }

  public LogNumber add(final double right) {
    return add(LogNumber.log(right));
  }

  public LogNumber add(final LogNumber right) {
    final LogNumber left = this;
    // assert left.isFinite();
    // assert right.isFinite();
    if (null == right) return left;
    if (left.logValue < right.logValue) return right.add(left);
    if (right.logValue - left.logValue < -LogNumber.ADDITION_PRECISION) return this;
    final LogNumber left2 = new LogNumber(left.type, 0.); // left.logValue-left.logValue
    final LogNumber right2 = new LogNumber(right.type, right.logValue - left.logValue);
    final LogNumber result = LogNumber.log(right2.doubleValue() + left2.doubleValue());
    // assert result.isFinite();
    final LogNumber result2 = new LogNumber(result.type, result.logValue + left.logValue);
    return result2;
  }

  @Override
  public int compareTo(final LogNumber o) {
    int compare = Byte.compare(this.type, o.type);
    if (0 == compare) {
      compare = Double.compare(this.type * this.logValue, o.type * o.logValue);
    }
    if (0 == compare) {
      compare = Double.compare(System.identityHashCode(this), System.identityHashCode(o));
    }
    return compare;
  }

  public LogNumber divide(final double right) {
    return divide(LogNumber.log(right));
  }

  public LogNumber divide(final LogNumber right) {
    if (null == right) return this;
    if (0 == right.type) return new LogNumber(this.type, Double.NaN);
    final LogNumber r = new LogNumber((byte) (this.type * right.type), this.logValue - right.logValue);
    return r;
  }
  
  @Override
  public double doubleValue() {
    final double exp = 0. == this.logValue ? 1. : Math.exp(this.logValue);
    if (Double.isNaN(this.logValue) && !Double.isFinite(exp))
      if ((byte) 0 == this.type)
      return Double.NaN;
    else if ((byte) 1 == this.type)
      return Double.POSITIVE_INFINITY;
    else return Double.NEGATIVE_INFINITY;
    return 0 == this.type ? 0 : this.type * exp;
  }

  @Override
  public float floatValue() {
    return (float) doubleValue();
  }

  @Override
  public int intValue() {
    return (int) doubleValue();
  }

  public boolean isFinite() {
    return Double.isFinite(this.logValue);
  }

  public boolean isNegative() {
    return this.type == -1;
  }
  
  @Override
  public long longValue() {
    return (long) doubleValue();
  }
  
  public LogNumber multiply(final double right) {
    return multiply(LogNumber.log(right));
  }
  
  public LogNumber multiply(final LogNumber right) {
    if (null == right) return this;
    assert isFinite();
    // assert right.isFinite();
    final LogNumber r = new LogNumber((byte) (this.type * right.type), this.logValue + right.logValue);
    // assert r.isFinite();
    return r;
  }
  
  private LogNumber negate() {
    return new LogNumber((byte) -this.type, this.logValue);
  }
  
  public LogNumber subtract(final double right) {
    return subtract(LogNumber.log(right));
  }
  
  public LogNumber subtract(final LogNumber value) {
    if (null == value) return this;
    return add(value.negate());
  }
  
  @Override
  public String toString() {
    final StringBuilder builder = new StringBuilder();
    builder.append(doubleValue());
    return builder.toString();
  }
  
}