package com.simiacryptus.mindseye.net.basic;

import java.util.List;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.NNResult;
import com.simiacryptus.mindseye.net.NNLayer;

import groovy.lang.Tuple2;

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

  @Override
  public List<Tuple2<Integer, Integer>> permuteInput(final List<Tuple2<Integer, Integer>> permute) {
    return super.permuteInput(permute);
  }

  @Override
  public List<Tuple2<Integer, Integer>> permuteOutput(final List<Tuple2<Integer, Integer>> permute) {
    return super.permuteOutput(permute);
  }

  public final void setInner(final NNLayer<?> inner) {
    this.inner = inner;
  }

  @Override
  public List<double[]> state() {
    return getInner().state();
  }

}
