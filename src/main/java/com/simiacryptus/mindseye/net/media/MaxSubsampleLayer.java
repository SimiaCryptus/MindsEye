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

package com.simiacryptus.mindseye.net.media;

import com.simiacryptus.lang.Tuple2;
import com.simiacryptus.mindseye.net.DeltaSet;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.NNResult;
import com.simiacryptus.util.ml.Coordinate;
import com.simiacryptus.util.ml.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;
import java.util.function.ToDoubleFunction;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class MaxSubsampleLayer extends NNLayer {
  private static final Function<CalcRegionsParameter, List<Tuple2<Coordinate, List<Coordinate>>>> calcRegionsCache = ConvolutionSynapseLayer
                                                                                                                         .cache(MaxSubsampleLayer::calcRegions);
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(MaxSubsampleLayer.class);
  /**
   *
   */
  private static final long serialVersionUID = -4486788592198117530L;
  private int[] kernelDims;
  
  protected MaxSubsampleLayer() {
    super();
  }
  
  public MaxSubsampleLayer(final int... kernelDims) {
    
    this.kernelDims = Arrays.copyOf(kernelDims, kernelDims.length);
  }
  
  private static List<Tuple2<Coordinate, List<Coordinate>>> calcRegions(final CalcRegionsParameter p) {
    final Tensor input = new Tensor(p.inputDims);
    final int[] newDims = IntStream.range(0, p.inputDims.length).map(i -> {
      //assert 0 == p.inputDims[i] % p.kernelDims[i];
      return (int) Math.ceil(p.inputDims[i] * 1.0 / p.kernelDims[i]);
    }).toArray();
    final Tensor output = new Tensor(newDims);
    
    return output.coordStream(false).map(o -> {
      final List<Coordinate> inCoords = new Tensor(p.kernelDims).coordStream(false).map(kernelCoord -> {
        final int[] result = new int[o.coords.length];
        for (int index = 0; index < o.coords.length; index++) {
          int outputCoordinate = o.coords[index];
          int kernelSize = p.kernelDims[index];
          int baseCoordinate = Math.min(outputCoordinate * kernelSize, p.inputDims[index] - kernelSize);
          int kernelCoordinate = kernelCoord.coords[index];
          result[index] = baseCoordinate + kernelCoordinate;
        }
        return new Coordinate(input.index(result), result);
      }).collect(Collectors.toList());
      return new Tuple2<>(o, inCoords);
    }).collect(Collectors.toList());
  }
  
  @Override
  public NNResult eval(final NNResult... inObj) {
    
    int itemCnt = inObj[0].data.length;
    @SuppressWarnings("unchecked") final ConcurrentHashMap<Coordinate, Coordinate> gradientMapA[] = new ConcurrentHashMap[itemCnt];
    
    final int[] inputDims = inObj[0].data[0].getDims();
    Tensor[] outputA = IntStream.range(0, inObj[0].data.length).mapToObj(dataIndex -> {
      final Tensor input = inObj[0].data[dataIndex];
      final int[] newDims = IntStream.range(0, inputDims.length).map(i -> {
        //assert 0 == inputDims[i] % this.kernelDims[i];
        return (int) Math.ceil(inputDims[i] * 1.0 / this.kernelDims[i]);
      }).toArray();
      final Tensor output = new Tensor(newDims);
      final ConcurrentHashMap<Coordinate, Coordinate> gradientMap = new ConcurrentHashMap<Coordinate, Coordinate>();
      final List<Tuple2<Coordinate, List<Coordinate>>> regions = calcRegionsCache.apply(new CalcRegionsParameter(inputDims, this.kernelDims));
      final ToDoubleFunction<? super Coordinate> keyExtractor = inputCoords -> input.get(inputCoords);
      regions.forEach(tuple -> {
        final Coordinate from = tuple.getFirst();
        List<Coordinate> toList = tuple.getSecond();
        Coordinate toMax = null;
        double bestValue = Double.NEGATIVE_INFINITY;
        for(Coordinate c : toList) {
          double value = keyExtractor.applyAsDouble(c);
          if(null == toMax || bestValue < value) {
            bestValue = value;
            toMax = c;
          }
        }
        gradientMap.put(from, toMax);
        output.set(from, input.get(toMax));
      });
      gradientMapA[dataIndex] = gradientMap;
      return output;
    }).toArray(i -> new Tensor[i]);
    return new NNResult(outputA) {
      @Override
      public void accumulate(final DeltaSet buffer, final Tensor[] data) {
        if (inObj[0].isAlive()) {
          Tensor[] passbackA = IntStream.range(0, inObj[0].data.length).mapToObj(dataIndex -> {
            final Tensor backSignal = new Tensor(inputDims);
            gradientMapA[dataIndex].entrySet().forEach(e -> backSignal.add(e.getValue().index, data[dataIndex].get(e.getKey().index)));
            return backSignal;
          }).toArray(i -> new Tensor[i]);
          inObj[0].accumulate(buffer, passbackA);
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
  
  public static class CalcRegionsParameter {
    public int[] inputDims;
    public int[] kernelDims;
    
    public CalcRegionsParameter(final int[] inputDims, final int[] kernelDims) {
      this.inputDims = inputDims;
      this.kernelDims = kernelDims;
    }
    
    @Override
    public boolean equals(final Object obj) {
      if (this == obj)
        return true;
      if (obj == null)
        return false;
      if (getClass() != obj.getClass())
        return false;
      final CalcRegionsParameter other = (CalcRegionsParameter) obj;
      if (!Arrays.equals(this.inputDims, other.inputDims))
        return false;
      return Arrays.equals(this.kernelDims, other.kernelDims);
    }
    
    @Override
    public int hashCode() {
      final int prime = 31;
      int result = 1;
      result = prime * result + Arrays.hashCode(this.inputDims);
      result = prime * result + Arrays.hashCode(this.kernelDims);
      return result;
    }
    
  }
}
