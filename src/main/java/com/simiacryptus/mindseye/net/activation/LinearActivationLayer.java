package com.simiacryptus.mindseye.net.activation;

import java.util.Arrays;
import java.util.List;
import java.util.function.DoubleSupplier;

import com.simiacryptus.util.ml.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.Util;
import com.simiacryptus.mindseye.core.delta.DeltaSet;
import com.simiacryptus.mindseye.core.delta.NNLayer;
import com.simiacryptus.mindseye.core.delta.NNResult;

public class LinearActivationLayer extends NNLayer<LinearActivationLayer> {
  private final class Result extends NNResult {
    private final NNResult inObj;

    private Result(final Tensor[] outputA, final NNResult inObj) {
      super(outputA);
      this.inObj = inObj;
    }

    @Override
    public void accumulate(final DeltaSet buffer, final Tensor[] delta) {

      if (!isFrozen()) {
        java.util.stream.IntStream.range(0, delta.length).forEach(dataIndex->{
          final double[] deltaData = delta[dataIndex].getData();
          final double[] inputData = this.inObj.data[dataIndex].getData();
          final Tensor weightDelta = new Tensor(LinearActivationLayer.this.weights.getDims());
          for (int i = 0; i < deltaData.length; i++) {
            weightDelta.add(0, deltaData[i] * inputData[i]);
          }
          buffer.get(LinearActivationLayer.this, LinearActivationLayer.this.weights).accumulate(weightDelta.getData());
        });
      }
      if (this.inObj.isAlive()) {
        Tensor[] passbackA = java.util.stream.IntStream.range(0, delta.length).mapToObj(dataIndex->{
          final double[] deltaData = delta[dataIndex].getData();
          final int[] dims = this.inObj.data[dataIndex].getDims();
          final Tensor passback = new Tensor(dims);
          for (int i = 0; i < passback.dim(); i++) {
            passback.set(i, deltaData[i] * LinearActivationLayer.this.weights.getData()[0]);
          }
          return passback;
        }).toArray(i->new Tensor[i]);
        this.inObj.accumulate(buffer, passbackA);
      }
    }

    @Override
    public boolean isAlive() {
      return this.inObj.isAlive() || !isFrozen();
    }

  }

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(LinearActivationLayer.class);

  /**
   * 
   */
  private static final long serialVersionUID = -2105152439043901220L;

  private final Tensor weights;

  public LinearActivationLayer() {
    super();
    this.weights = new Tensor(1);
    this.weights.set(0, 1.);
  }

  public LinearActivationLayer addWeights(final DoubleSupplier f) {
    Util.add(f, this.weights.getData());
    return this;
  }

  @Override
  public NNResult eval(final NNResult... inObj) {
    int itemCnt = inObj[0].data.length;
    Tensor[] outputA = java.util.stream.IntStream.range(0, itemCnt).mapToObj(dataIndex->{
      final Tensor input = inObj[0].data[dataIndex];
      final double a = this.weights.get(0);
      final Tensor output = input.scale(a);
      return output;
    }).toArray(i->new Tensor[i]);
    return new Result(outputA, inObj[0]);
  }

  @Override
  public JsonObject getJson() {
    final JsonObject json = super.getJson();
    json.addProperty("weights", this.weights.toString());
    return json;
  }

  protected double getMobility() {
    return 1;
  }

  public LinearActivationLayer setWeight(final double data) {
    this.weights.set(0,data);
    return this;
  }

  public LinearActivationLayer setWeights(final DoubleSupplier f) {
    Arrays.parallelSetAll(this.weights.getData(), i -> f.getAsDouble());
    return this;
  }

  @Override
  public List<double[]> state() {
    return Arrays.asList(this.weights.getData());
  }

}
