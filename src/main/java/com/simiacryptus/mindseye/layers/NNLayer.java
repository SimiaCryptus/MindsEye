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

  private final String id = UUID.randomUUID().toString();

  public final NNResult eval(final EvaluationContext evaluationContext, final NDArray... array) {
    return eval(evaluationContext, NNLayer.wrapInput(array));
  }

  public abstract NNResult eval(EvaluationContext evaluationContext, NNResult... array);

  public List<NNLayer> getChildren() {
    return Arrays.asList(this);
  }

  public String getId() {
    return this.id;
  }

  public JsonObject getJson() {
    final JsonObject json = new JsonObject();
    json.addProperty("class", getClass().getSimpleName());
    json.addProperty("id", getId());
    return json;
  }

  public abstract List<double[]> state();

  public boolean isVerbose() {
    return false;
  }

  @Override
  public final String toString() {
    return new GsonBuilder().setPrettyPrinting().create().toJson(getJson());
  }

}
