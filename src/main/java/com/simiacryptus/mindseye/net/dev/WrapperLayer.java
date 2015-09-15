package com.simiacryptus.mindseye.net.dev;

import java.util.List;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.deltas.NNResult;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.dag.EvaluationContext;

import groovy.lang.Tuple2;

public final class WrapperLayer extends NNLayer {
  private NNLayer inner;

  public WrapperLayer(NNLayer inner) {
    super();
    this.setInner(inner);
  }

  @Override
  public NNResult eval(EvaluationContext evaluationContext, NNResult... array) {
    return getInner().eval(evaluationContext, array);
  }

  @Override
  public List<double[]> state() {
    return getInner().state();
  }

  @Override
  public List<NNLayer> getChildren() {
    return super.getChildren();
  }

  @Override
  public JsonObject getJson() {
    return super.getJson();
  }

  @Override
  public boolean isVerbose() {
    return super.isVerbose();
  }

  @Override
  public List<Tuple2<Integer, Integer>> permuteInput(List<Tuple2<Integer, Integer>> permute) {
    return super.permuteInput(permute);
  }

  @Override
  public List<Tuple2<Integer, Integer>> permuteOutput(List<Tuple2<Integer, Integer>> permute) {
    return super.permuteOutput(permute);
  }

  public final NNLayer getInner() {
    return inner;
  }

  public final void setInner(NNLayer inner) {
    this.inner = inner;
  }
  
}
