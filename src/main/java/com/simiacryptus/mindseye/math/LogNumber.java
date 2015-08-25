package com.simiacryptus.mindseye.math;

@SuppressWarnings("serial")
public class LogNumber extends Number implements Comparable<LogNumber> {
  public static final LogNumber zero = new LogNumber((byte) 0,0);
  private static final LogNumber NaN = new LogNumber((byte) 0,Double.NaN);
  public final double logValue;
  public final byte type;

  private LogNumber(byte type, double logValue) {
    super();
    this.type = type;
    this.logValue = logValue;
  }

  private LogNumber(boolean realNeg, double logValue) {
    super();
    this.type = (byte) (realNeg?-1:1);
    this.logValue = logValue;
  }
  
  public static LogNumber log(double v) {
    if(0. == Math.abs(v)) return new LogNumber((byte) 0, 0);
    LogNumber logNumber = new LogNumber((byte) (v < 0. ? -1 : 1), Math.log(Math.abs(v)));
    assert(Math.abs(Math.log(logNumber.doubleValue()/v)) < 0.001);
    return logNumber;
  }
  
  @Override
  public int intValue() {
    return (int) doubleValue();
  }
  
  @Override
  public long longValue() {
    return (long) doubleValue();
  }
  
  @Override
  public float floatValue() {
    return (float) doubleValue();
  }
  
  @Override
  public double doubleValue() {
    double exp = 0.==logValue?1.:Math.exp(logValue);
    if(Double.isNaN(logValue) && !Double.isFinite(exp)) 
      if(((byte)0) == type){
        return Double.NaN;
      } else if(((byte)1) == type){
        return Double.POSITIVE_INFINITY;
      } else {
        return Double.NEGATIVE_INFINITY;
      }
    return 0==type?0:(type * exp);
  }
  
  @Override
  public int compareTo(LogNumber o) {
    int compare = Byte.compare(this.type, o.type);
    if (0 == compare) compare = Double.compare(this.logValue, o.logValue);
    return compare;
  }
  
  public LogNumber add(LogNumber right) {
    LogNumber left = this;
    assert (left.isFinite());
    assert (right.isFinite());
    if(null == right) return left;
    if(left.logValue < right.logValue) return right.add(left);
    if((right.logValue-left.logValue) < -8) return this;
    LogNumber left2 = new LogNumber(left.type, 0.); // left.logValue-left.logValue
    LogNumber right2 = new LogNumber(right.type, right.logValue-left.logValue);
    LogNumber result = LogNumber.log(right2.doubleValue() + left2.doubleValue());
    assert (result.isFinite());
    LogNumber result2 = new LogNumber(result.type, result.logValue + left.logValue);
    return result2;
  }

  public LogNumber multiply(double right) {
    return multiply(log(right));
  }
  
  public LogNumber add(double right) {
    return add(log(right));
  }
  
  public LogNumber divide(double right) {
    return divide(log(right));
  }
  
  public LogNumber subtract(double right) {
    return subtract(log(right));
  }
  
  public LogNumber multiply(LogNumber right) {
    if(null == right) return this;
    assert (this.isFinite());
    assert (right.isFinite());
    LogNumber r = new LogNumber((byte) (type * right.type), logValue + right.logValue);
    assert (r.isFinite());
    return r;
  }

  public boolean isFinite() {
    return Double.isFinite(logValue);
  }

  public LogNumber subtract(LogNumber value) {
    if(null == value) return this;
    return add(value.negate());
  }

  private LogNumber negate() {
    return new LogNumber((byte) -type, logValue);
  }

  public LogNumber divide(LogNumber right) {
    if(null == right) return this;
    if(0 == right.type) return new LogNumber(type, Double.NaN);
    LogNumber r = new LogNumber((byte) (type * right.type), logValue - right.logValue);
    return r;
  }

  public boolean isNegative() {
    return type==-1;
  }

  @Override
  public String toString() {
    StringBuilder builder = new StringBuilder();
    builder.append(doubleValue());
//    builder.append(type);
//    builder.append("*e^");
//    builder.append(logValue);
    return builder.toString();
  }

  
  
}