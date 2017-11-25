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

import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.layers.cudnn.CudaExecutionContext;

import java.util.Arrays;
import java.util.concurrent.Callable;
import java.util.stream.IntStream;

public class SimpleEval implements Callable<SimpleEval> {
  private NNLayer layer;
  private Tensor[] input;
  private Tensor[] derivative;
  private Tensor output;
  
  public SimpleEval(NNLayer layer, Tensor... input) {
    this.layer = layer;
    this.input = input;
  }
  
  public Tensor[] getDerivative() {
    return derivative;
  }
  
  public Tensor getOutput() {
    return output;
  }
  
  @Override
  public SimpleEval call() {
    derivative = Arrays.stream(input).map(input->new Tensor(input.getDimensions())).toArray(i->new Tensor[i]);
    NNResult[] inputR = IntStream.range(0,input.length).mapToObj(i->{
      return new NNResult(input[i]) {
        @Override
        public void accumulate(DeltaSet buffer, TensorList data) {
          derivative[i].accum(data.get(0));
        }
    
        @Override
        public boolean isAlive() {
          return true;
        }
      };
    }).toArray(i->new NNResult[i]);
    NNResult result = CudaExecutionContext.gpuContexts.run(cudaExeCtx -> {
      NNResult eval = layer.eval(cudaExeCtx, inputR);
      eval.accumulate(new DeltaSet());
      return eval;
    });
    output = result.getData().get(0);
    return this;
  }
  
  public static SimpleEval run(NNLayer layer, Tensor... tensor) {
    return new SimpleEval(layer, tensor).call();
  }
}
