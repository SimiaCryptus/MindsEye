package com.simiacryptus.mindseye;

import java.util.Random;

import java.util.stream.IntStream;

import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class TestJava {
  
  private static final Logger log = LoggerFactory.getLogger(TestJava.class);
  
  @Test
  public void testTightStreamLoop() throws Exception {
    {
      final int[] x = new int[1000000];
      test(() -> {
        for (int i = 0; i < x.length; i++)
          x[i] += 1;
      });
    }
    {
      final int[] x = new int[1000000];
      test(() -> {
        IntStream.range(0, x.length).forEach(i -> x[i] += 1);
      });
    }
    {
      final int[] x = new int[1000000];
      test(() -> {
        IntStream.range(0, x.length).parallel().forEach(i -> x[i] += 1);
      });
    }
  }
  
  private void test(Runnable task) {
    test(task, Thread.currentThread().getStackTrace()[2].toString());
  }
  
  private void test(Runnable task, String label) {
    long start = System.nanoTime();
    task.run();
    double duration = (System.nanoTime() - start) * 1.0e-9;
    log.debug(String.format("Tested %s in %.05fs", label, duration));
  }
  
}
