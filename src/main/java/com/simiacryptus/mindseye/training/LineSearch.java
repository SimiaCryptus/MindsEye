package com.simiacryptus.mindseye.training;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public abstract class LineSearch {

  private static final Logger log = LoggerFactory.getLogger(LineSearch.class);

  private Object currentPos;

  public double run() {
    double phi = (Math.sqrt(5)-1)/2;
    
    double y1 = eval(0);
    double x1 = 0;
    log.debug(String.format("Evaluating initial position, error %s", y1));
    
    double x2 = 0.5;
    double errOuterB;
    do {
      x2 *= 2;
      if(x2 > 1000) {
        currentPos = eval(x2);
        log.debug(String.format("Undefined outer bounds"));
        return Double.POSITIVE_INFINITY;
      }
      errOuterB = eval(x2);
      log.debug(String.format("Evaluating initial outer %s, error %s", currentPos, errOuterB));
    } while(errOuterB <= y1);
    
    double windowStopSize = 0.001;
    
    double x3 = x2 + phi * (x1 - x2);
    double y3 = eval(x3);
    log.debug(String.format("Evaluating initial inner A: %s, error %s", x3, y3));

    double x4 = x1 + phi * (x2 - x1);
    double y4 = eval(x4);
    log.debug(String.format("Evaluating initial inner B: %s, error %s", x4, y4));

    while(Math.abs(x1 - x2) > windowStopSize) {
      if(y3 < y4) {
        x2 = x4;
        errOuterB = y4;
        
        x4 = x2 - phi * (x2 - x1);
        y4 = eval(x4);
        log.debug(String.format("Evaluating new inner B: %s, error %s; pos=%s,%s,%s", x4, y4, x1, x3, x2));
      } else {
        x1 = x3;
        y1 = y3;
        
        x3 = x1 - phi * (x1 - x2);
        y3 = eval(x3);
        log.debug(String.format("Evaluating new inner A: %s, error %s; pos=%s,%s,%s", x3, y3, x1, x4, x2));
      }
    }
    return x1;
  }

  protected abstract double eval(double x);
  
}
