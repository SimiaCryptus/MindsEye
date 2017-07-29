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

package com.simiacryptus.mindseye.layers.cross;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.layers.DeltaSet;
import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.layers.NNResult;
import com.simiacryptus.util.ml.Tensor;

import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

public class CrossProductLayer extends NNLayer {
  
  public JsonObject getJson() {
    return super.getJsonStub();
  }
  public static CrossProductLayer fromJson(JsonObject json) {
    return new CrossProductLayer(json);
  }
  protected CrossProductLayer(JsonObject id) {
    super(id);
  }

  public CrossProductLayer() {
  }
  
  @Override
  public NNResult eval(final NNResult... inObj) {
    assert(1 == inObj.length);
    return new NNResult(inObj[0].data.stream().parallel().map(tensor->{
      int inputDim = tensor.dim();
      int outputDim = (inputDim * inputDim - inputDim) / 2;
      Tensor result = new Tensor(outputDim);
      double[] inputData = tensor.getData();
      double[] resultData = result.getData();
      IntStream.range(0, inputDim).forEach(x->{
        IntStream.range(x+1, inputDim).forEach(y->{
          resultData[index(x,y,inputDim)] = inputData[x] * inputData[y];
        });
      });
      return result;
    }).toArray(i->new Tensor[i])) {
      @Override
      public void accumulate(final DeltaSet buffer, final Tensor[] data) {
        final NNResult input = inObj[0];
        if (input.isAlive()) {
          assert(inObj[0].data.length() ==data.length);
          input.accumulate(buffer, IntStream.range(0, inObj[0].data.length()).parallel().mapToObj(batchIndex->{
            Tensor tensor = data[batchIndex];
            int outputDim = tensor.dim();
            int inputDim = (1+(int)Math.sqrt(1+8 * outputDim))/2;
            Tensor passback = new Tensor(inputDim);
            double[] passbackData = passback.getData();
            double[] tensorData = tensor.getData();
            double[] inputData = inObj[0].data.get(batchIndex).getData();
            IntStream.range(0, inputDim).forEach(x->{
              IntStream.range(x+1, inputDim).forEach(y->{
                passbackData[x] += tensorData[index(x,y,inputDim)] * inputData[y];
                passbackData[y] += tensorData[index(x,y,inputDim)] * inputData[x];
              });
            });
            return passback;
          }).toArray(i->new Tensor[i]));
        }
      }
      
      @Override
      public boolean isAlive() {
        for (final NNResult element : inObj)
          if (element.isAlive())
            return true;
        return false;
      }
      
    };
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
  
  public static int index(int x, int y, int max) {
    return (max *(max -1)/2) - (max - x)*((max - x)-1)/2 + y - x - 1;
  }
  
}
