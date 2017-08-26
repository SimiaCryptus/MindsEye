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

package com.simiacryptus.mindseye.layers.loss;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.layers.*;
import com.simiacryptus.util.ml.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

/**
 * The type Entropy loss layer.
 */
public class EntropyLossLayer extends NNLayer {
  
  public JsonObject getJson() {
    return super.getJsonStub();
  }

  /**
   * From json entropy loss layer.
   *
   * @param json the json
   * @return the entropy loss layer
   */
  public static EntropyLossLayer fromJson(JsonObject json) {
    return new EntropyLossLayer(json);
  }

  /**
   * Instantiates a new Entropy loss layer.
   *
   * @param id the id
   */
  protected EntropyLossLayer(JsonObject id) {
    super(id);
  }
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(EntropyLossLayer.class);
  /**
   *
   */
  private static final long serialVersionUID = -6257785994031662519L;
  
  /**
   * Instantiates a new Entropy loss layer.
   */
  public EntropyLossLayer() {
  }
  
  @Override
  public NNResult eval(NNExecutionContext nncontext, final NNResult... inObj) {
    Tensor gradientA[] = new Tensor[inObj[0].getData().length()];
    Tensor[] outputA = IntStream.range(0, inObj[0].getData().length()).mapToObj(dataIndex -> {
      final Tensor l = inObj[0].getData().get(dataIndex);
      final Tensor r = inObj[1].getData().get(dataIndex);
      assert (l.dim() == r.dim()) : l.dim() + " != " + r.dim();
      final Tensor gradient = new Tensor(l.getDimensions());
      gradientA[dataIndex] = gradient;
      final double[] gradientData = gradient.getData();
      final double descriptiveNats;
      double total = 0;
      double[] ld = l.getData();
      double[] rd = r.getData();
      for (int i = 0; i < l.dim(); i++) {
        final double lv = Math.max(Math.min(ld[i], 1.), 1e-12);
        final double rv = rd[i];
        if(rv > 0) {
          gradientData[i] = -rv / lv;
          total += -rv * Math.log(lv);
        } else {
          gradientData[i] = 0;
        }
      }
      assert(total >= 0);
      descriptiveNats = total;
      
      return new Tensor(new int[]{1}, new double[]{descriptiveNats});
    }).toArray(i -> new Tensor[i]);
    return new NNResult(outputA) {
      @Override
      public void accumulate(final DeltaSet buffer, final TensorList data) {
        NNResult a = inObj[0];
        NNResult b = inObj[1];
        if (b.isAlive()) {
          throw new RuntimeException();
        }
        if (a.isAlive()) {
          a.accumulate(buffer, new TensorArray(IntStream.range(0, data.length()).mapToObj(dataIndex -> {
            final Tensor passback = new Tensor(gradientA[dataIndex].getDimensions());
            for (int i = 0; i < a.getData().get(0).dim(); i++) {
              passback.set(i, data.get(dataIndex).get(0) * gradientA[dataIndex].get(i));
            }
            return passback;
          }).toArray(i -> new Tensor[i])));
        }
      }
      
      @Override
      public boolean isAlive() {
        return inObj[0].isAlive();
      }
      
    };
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
}
