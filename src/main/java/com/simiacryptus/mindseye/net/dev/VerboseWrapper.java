package com.simiacryptus.mindseye.net.dev;

import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.NNResult;
import com.simiacryptus.mindseye.net.NNLayer;

@SuppressWarnings("serial")
public final class VerboseWrapper extends NNLayer<VerboseWrapper> {

  static final Logger log = LoggerFactory.getLogger(VerboseWrapper.class);

  private final NNLayer<?> inner;
  private final String label;

  public VerboseWrapper(String label, NNLayer<?> inner) {
    this.inner = inner;
    this.label = label;
  }

  @Override
  public NNResult eval(NNResult... inObj) {
    NNResult result = inner.eval(inObj);
    log.debug(String.format("%s: %s => %s", label, java.util.Arrays.stream(inObj).map(l->l.data).collect(java.util.stream.Collectors.toList()), result.data));
    return result;
  }

  @Override
  public List<double[]> state() {
    return inner.state();
  }
}