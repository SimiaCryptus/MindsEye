package com.simiacryptus.mindseye.net.util;

import java.util.List;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.core.delta.NNLayer;
import com.simiacryptus.mindseye.core.delta.NNResult;

public class WrapperLayer extends NNLayer<WrapperLayer> {
  /**
   * 
   */
  private static final long serialVersionUID = 6284058717982209085L;
  private NNLayer<?> inner;

  public WrapperLayer(final NNLayer<?> inner) {
    super();
    setInner(inner);
  }

  @Override
  public NNResult eval(final NNResult... array) {
    return getInner().eval(array);
  }

  @Override
  public List<NNLayer<?>> getChildren() {
    return super.getChildren();
  }

  public final NNLayer<?> getInner() {
    return this.inner;
  }

  @Override
  public JsonObject getJson() {
    return this.inner.getJson();
  }

  public final void setInner(final NNLayer<?> inner) {
    this.inner = inner;
  }

  @Override
  public List<double[]> state() {
    return getInner().state();
  }

}
