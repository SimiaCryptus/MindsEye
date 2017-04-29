package com.simiacryptus.mindseye.test.demo;

import com.aparapi.Kernel;
import com.aparapi.Range;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Random;

public class AparapiTest {
  static final Logger log = LoggerFactory.getLogger(AparapiTest.class);

  public static final Random random = new Random();

  @Test
  public void test() throws Exception {
    float inA[] = new float[1024];
    float inB[] = new float[1024];
    assert (inA.length == inB.length);
    float[] result = new float[inA.length];

    Kernel kernel = new Kernel() {
      @Override
      public void run() {
        int i = getGlobalId();
        result[i] = inA[i] + inB[i];
      }
    };

    Range range = Range.create(result.length);
    kernel.execute(range);
  }


}
