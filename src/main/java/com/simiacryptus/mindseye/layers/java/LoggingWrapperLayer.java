/*
 * Copyright (c) 2018 by Andrew Charneski.
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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.Map;
import java.util.stream.IntStream;

/**
 * This wrapper adds a highly verbose amount of logging to System.out detailing all inputs and outputs during forward
 * and backwards evaluation. Intended as a diagnostic and demonstration tool.
 */
@SuppressWarnings("serial")
public final class LoggingWrapperLayer extends WrapperLayer {
  /**
   * The Logger.
   */
  static final Logger log = LoggerFactory.getLogger(LoggingWrapperLayer.class);
  
  
  /**
   * Instantiates a new Monitoring wrapper layer.
   *
   * @param json the json
   */
  protected LoggingWrapperLayer(final JsonObject json) {
    super(json);
  }
  
  /**
   * Instantiates a new Monitoring wrapper layer.
   *
   * @param inner the inner
   */
  public LoggingWrapperLayer(final NNLayer inner) {
    super(inner);
  }
  
  /**
   * From json monitoring wrapper layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the monitoring wrapper layer
   */
  public static LoggingWrapperLayer fromJson(final JsonObject json, Map<String, byte[]> rs) {
    return new LoggingWrapperLayer(json);
  }
  
  @Override
  public NNResult eval(final NNExecutionContext nncontext, final NNResult... inObj) {
    final NNResult[] wrappedInput = IntStream.range(0, inObj.length).mapToObj(i -> {
      final NNResult result = inObj[i];
      return new NNResult(result.getData()) {
  
        @Override
        public void finalize() {
          Arrays.stream(inObj).forEach(NNResult::finalize);
        }
  
        @Override
        public void accumulate(final DeltaSet<NNLayer> buffer, final TensorList data) {
          final String formatted = data.stream().map(x -> x.prettyPrint())
                                       .reduce((a, b) -> a + "\n" + b).get();
          log.info(String.format("Feedback Output %s for layer %s: \n\t%s", i, getInner().getName(), formatted.replaceAll("\n", "\n\t")));
          result.accumulate(buffer, data);
        }
        
        @Override
        public boolean isAlive() {
          return result.isAlive();
        }
      };
    }).toArray(i -> new NNResult[i]);
    for (int i = 0; i < inObj.length; i++) {
      final TensorList tensorList = inObj[i].getData();
      final String formatted = tensorList.stream().map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).get();
      log.info(String.format("Input %s for layer %s: \n\t%s", i, getInner().getName(), formatted.replaceAll("\n", "\n\t")));
    }
    final NNResult output = getInner().eval(nncontext, wrappedInput);
    
    {
      final TensorList tensorList = output.getData();
      final String formatted = tensorList.stream().map(x -> x.prettyPrint())
                                         .reduce((a, b) -> a + "\n" + b).get();
      log.info(String.format("Output for layer %s: \n\t%s", getInner().getName(), formatted.replaceAll("\n", "\n\t")));
    }
    
    return new NNResult(output.getData()) {
  
      @Override
      public void finalize() {
        Arrays.stream(inObj).forEach(NNResult::finalize);
      }
  
      @Override
      public void accumulate(final DeltaSet<NNLayer> buffer, final TensorList data) {
        final String formatted = data.stream().map(x -> x.prettyPrint())
                                     .reduce((a, b) -> a + "\n" + b).get();
        log.info(String.format("Feedback Input for layer %s: \n\t%s", getInner().getName(), formatted.replaceAll("\n", "\n\t")));
        output.accumulate(buffer, data);
      }
      
      @Override
      public boolean isAlive() {
        return output.isAlive();
      }
    };
  }
  
}
