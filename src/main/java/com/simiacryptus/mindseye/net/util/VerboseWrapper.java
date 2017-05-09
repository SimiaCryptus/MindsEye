package com.simiacryptus.mindseye.net.util;

import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.NNResult;

@SuppressWarnings("serial")
public final class VerboseWrapper extends NNLayer {

  static final Logger log = LoggerFactory.getLogger(VerboseWrapper.class);

  public final NNLayer inner;
  public final String label;

  public VerboseWrapper(final String label, final NNLayer inner) {
    this.inner = inner;
    this.label = label;
  }

  @Override
  public NNResult eval(final NNResult... inObj) {
    final NNResult result = this.inner.eval(inObj);
    log.debug(String.format("%s: %s => %s", this.label, 
        java.util.Arrays.stream(inObj).map(l -> java.util.Arrays.toString(l.data)).collect(java.util.stream.Collectors.toList()), 
        java.util.Arrays.toString(result.data)));
    return result;
  }

  @Override
  public List<double[]> state() {
    return this.inner.state();
  }
}
