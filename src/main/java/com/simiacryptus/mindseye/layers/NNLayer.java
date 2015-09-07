package com.simiacryptus.mindseye.layers;

import java.util.Arrays;
import java.util.List;
import java.util.UUID;
import java.util.stream.Stream;

import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.deltas.DeltaBuffer;
import com.simiacryptus.mindseye.deltas.NNResult;
import com.simiacryptus.mindseye.math.LogNDArray;
import com.simiacryptus.mindseye.math.NDArray;
import com.simiacryptus.mindseye.training.EvaluationContext;

import groovy.lang.Tuple2;

/**
 * Nonlinear Network Layer (aka Neural Network Layer)
 *
 * @author Andrew Charneski
 */
public abstract class NNLayer {

  public static NNResult[] wrapInput(final NDArray... array) {
    return Stream.of(array).map(a -> new NNResult(a) {
      @Override
      public void feedback(final LogNDArray data, final DeltaBuffer buffer) {
        // Do Nothing
      }

      @Override
      public boolean isAlive() {
        return false;
      }
    }).toArray(i -> new NNResult[i]);
  }

  private final UUID id = UUID.randomUUID();

  @Override
  public boolean equals(final Object obj) {
    if (this == obj)
      return true;
    if (obj == null)
      return false;
    if (getClass() != obj.getClass())
      return false;
    final NNLayer other = (NNLayer) obj;
    if (this.id == null) {
      if (other.id != null)
        return false;
    } else if (!this.id.equals(other.id))
      return false;
    return true;
  }

  public final NNResult eval(final EvaluationContext evaluationContext, final NDArray... array) {
    return eval(evaluationContext, NNLayer.wrapInput(array));
  }

  public abstract NNResult eval(EvaluationContext evaluationContext, NNResult... array);

  public List<NNLayer> getChildren() {
    return Arrays.asList(this);
  }

  public UUID getId() {
    return this.id;
  }

  public JsonObject getJson() {
    final JsonObject json = new JsonObject();
    json.addProperty("class", getClass().getSimpleName());
    json.addProperty("id", getId().toString());
    return json;
  }

  @Override
  public int hashCode() {
    final int prime = 31;
    int result = 1;
    result = prime * result + (this.id == null ? 0 : this.id.hashCode());
    return result;
  }

  public boolean isVerbose() {
    return false;
  }

  public List<Tuple2<Integer, Integer>> permuteInput(final List<Tuple2<Integer, Integer>> permute) {
    throw new RuntimeException("Not Implemented: permuteOutput:" + this);
  }

  public List<Tuple2<Integer, Integer>> permuteOutput(final List<Tuple2<Integer, Integer>> permute) {
    throw new RuntimeException("Not Implemented: permuteOutput:" + this);
  }

  public abstract List<double[]> state();

  @Override
  public final String toString() {
    return new GsonBuilder().setPrettyPrinting().create().toJson(getJson());
  }

}
