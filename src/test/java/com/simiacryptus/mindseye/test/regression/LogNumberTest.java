package com.simiacryptus.mindseye.test.regression;

import java.util.stream.DoubleStream;

import org.junit.Assert;
import org.junit.Test;

import com.simiacryptus.mindseye.LogNumber;

public class LogNumberTest {

  @Test
  public void test() {
    double[] vals = new double[]{0., 0.0001, 0.1, 10, 1000};
    DoubleStream.of(vals).forEach(x->{
      DoubleStream.of(vals).forEach(y->{        
        test(x, y);
        test(-x, y);
        test(x, -y);
        test(-x, -y);
      });
    });
  }

  public void test(double a, double b) {
    if(a == -0.) return;
    if(b == -0.) return;
    Assert.assertEquals(a*b, LogNumber.log(a).multiply(b).doubleValue(), .01);
    Assert.assertEquals(a+b, LogNumber.log(a).add(b).doubleValue(), .01);
    Assert.assertEquals(a/b, LogNumber.log(a).divide(b).doubleValue(), .01);
    Assert.assertEquals(a-b, LogNumber.log(a).subtract(b).doubleValue(), .01);
  }
  
}
