/*
 * Copyright (c) 2017 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.layers.java;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.*;

import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

/**
 * A multi-purpose Nth-power exponential function.
 * Has reasonably efficient specialized (pure java) 
 * implementations of many common signed rational values, 
 * such as +/-0.5, +/-1.0, 2.0, etc
 */
@SuppressWarnings("serial")
public final class NthPowerActivationLayer extends NNLayer {
  
  private double power = 1.0;
  
  /**
   * Instantiates a new Nth power activation layer.
   */
  public NthPowerActivationLayer() {
  }
  
  /**
   * Instantiates a new Nth power activation layer.
   *
   * @param id the id
   */
  protected NthPowerActivationLayer(final JsonObject id) {
    super(id);
    power = id.get("power").getAsDouble();
  }
  
  /**
   * From json nth power activation layer.
   *
   * @param json the json
   * @return the nth power activation layer
   */
  public static NthPowerActivationLayer fromJson(final JsonObject json) {
    return new NthPowerActivationLayer(json);
  }
  
  private static void nthPower(final double power, final Tensor input, final double[] inputData, final double[] gradientData, final double[] outputData) {
    for (int i = 0; i < input.dim(); i++) {
      final double x = inputData[i];
      final boolean isZero = Math.abs(x) < 1e-20;
      double d = isZero ? 0.0 : power * Math.pow(x, power - 1);
      double f = isZero ? 0.0 : Math.pow(x, power);
      if (!Double.isFinite(d)) {
        d = 0.0;
      }
      if (!Double.isFinite(f)) {
        f = 0.0;
      }
      gradientData[i] = d;
      outputData[i] = f;
    }
  }
  
  private static void square(final Tensor input, final double[] inputData, final double[] gradientData, final double[] outputData) {
    for (int i = 0; i < input.dim(); i++) {
      final double x = inputData[i];
      gradientData[i] = 2 * x;
      outputData[i] = x * x;
    }
  }
  
  private static void squareRoot(final Tensor input, final double[] inputData, final double[] gradientData, final double[] outputData) {
    for (int i = 0; i < input.dim(); i++) {
      final double x = inputData[i];
      final boolean isZero = Math.abs(x) < 1e-20;
      final double power = 0.5;
      final double v = Math.pow(x, power);
      double d = isZero ? 0.0 : power / v;
      double f = isZero ? 0.0 : v;
      if (!Double.isFinite(d)) {
        d = 0.0;
      }
      if (!Double.isFinite(f)) {
        f = 0.0;
      }
      gradientData[i] = d;
      outputData[i] = f;
    }
  }
  
  private static void unity(final Tensor input, final double[] inputData, final double[] gradientData, final double[] outputData) {
    for (int i = 0; i < input.dim(); i++) {
      gradientData[i] = 0;
      outputData[i] = 1;
    }
  }
  
  @Override
  public NNResult eval(final NNExecutionContext nncontext, final NNResult... inObj) {
    final int itemCnt = inObj[0].getData().length();
    assert 0 < itemCnt;
    final Tensor inputGradientA[] = new Tensor[itemCnt];
    final Tensor[] outputA = IntStream.range(0, itemCnt).parallel().mapToObj(dataIndex -> {
      final Tensor input = inObj[0].getData().get(dataIndex);
      final Tensor output = new Tensor(inObj[0].getData().get(dataIndex).getDimensions());
      final Tensor gradient = new Tensor(input.dim());
      final double[] inputData = input.getData();
      final double[] gradientData = gradient.getData();
      final double[] outputData = output.getData();
      inputGradientA[dataIndex] = gradient;
      if (power == 2) {
        NthPowerActivationLayer.square(input, inputData, gradientData, outputData);
      }
      else if (power == 0.5) {
        NthPowerActivationLayer.squareRoot(input, inputData, gradientData, outputData);
      }
      else if (power == 0.0) {
        NthPowerActivationLayer.unity(input, inputData, gradientData, outputData);
      }
      else {
        NthPowerActivationLayer.nthPower(power, input, inputData, gradientData, outputData);
      }
      return output;
    }).toArray(i -> new Tensor[i]);
    return new NNResult(outputA) {
      @Override
      public void accumulate(final DeltaSet<NNLayer> buffer, final TensorList data) {
        if (inObj[0].isAlive()) {
          final Tensor[] passbackA = IntStream.range(0, itemCnt).parallel().mapToObj(dataIndex -> {
            final Tensor passback = new Tensor(data.get(dataIndex).getDimensions());
            final double[] gradientData = inputGradientA[dataIndex].getData();
            IntStream.range(0, passback.dim()).forEach(i -> {
              final double v = gradientData[i];
              if (Double.isFinite(v)) {
                passback.set(i, data.get(dataIndex).getData()[i] * v);
              }
            });
            return passback;
          }).toArray(i -> new Tensor[i]);
          inObj[0].accumulate(buffer, new TensorArray(passbackA));
        }
      }
      
      @Override
      public boolean isAlive() {
        return 0.0 != power && inObj[0].isAlive();
      }
    };
  }
  
  @Override
  public JsonObject getJson() {
    final JsonObject json = super.getJsonStub();
    json.addProperty("power", power);
    return json;
  }
  
  /**
   * Gets power.
   *
   * @return the power
   */
  public double getPower() {
    return power;
  }
  
  /**
   * Sets power.
   *
   * @param power the power
   * @return the power
   */
  public NthPowerActivationLayer setPower(final double power) {
    this.power = power;
    return this;
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
  
}
