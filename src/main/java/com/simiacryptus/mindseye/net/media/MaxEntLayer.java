package com.simiacryptus.mindseye.net.media;

import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.deltas.DeltaBuffer;
import com.simiacryptus.mindseye.deltas.NNResult;
import com.simiacryptus.mindseye.math.NDArray;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.basic.SimpleActivationLayer;

public class MaxEntLayer extends SimpleActivationLayer<MaxEntLayer> {

  @Override
  protected void eval(double x, double[] results) {
    final double minDeriv = 0;
    double log = Math.log(x);
    double d = -(1+log);
    double f = -x*log;
    assert Double.isFinite(d);
    assert minDeriv <= Math.abs(d);
    results[0] = f;
    results[1] = d;
  }
}
