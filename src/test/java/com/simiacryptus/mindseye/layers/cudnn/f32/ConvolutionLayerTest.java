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

package com.simiacryptus.mindseye.layers.cudnn.f32;

import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.layers.DerivativeTester;
import com.simiacryptus.mindseye.layers.EquivalencyTester;

/**
 * The type Convolution layer run.
 */
public class ConvolutionLayerTest extends F32LayerTestBase {
  
  @Override
  public NNLayer getLayer() {
    return new ConvolutionLayer(3, 3, 1);
  }
  
  @Override
  public NNLayer getReferenceLayer() {
    return new com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer(3, 3, 1, 1, true);
  }
  
  @Override
  public int[][] getInputDims() {
    return new int[][]{
      {3, 3, 1}
    };
  }
  
  public static class AsymmetricTest extends ConvolutionLayerTest {
    
    @Override
    public NNLayer getLayer() {
      return new ConvolutionLayer(3, 3, 2, 4);
    }
    
    @Override
    public NNLayer getReferenceLayer() {
      return new com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer(3, 3, 2, 4, true);
    }
    
    public EquivalencyTester getEquivalencyTester() {
      return new EquivalencyTester(2);
    }
    
    @Override
    public int[][] getInputDims() {
      return new int[][]{
        {3, 3, 2}
      };
    }
    
  }
  
  public static class IrregularTest extends ConvolutionLayerTest {
    
    @Override
    public NNLayer getLayer() {
      return new ConvolutionLayer(3, 3, 2, 5);
    }
    
    @Override
    public NNLayer getReferenceLayer() {
      return new com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer(3, 3, 2, 5, true);
    }
    
    public EquivalencyTester getEquivalencyTester() {
      return new EquivalencyTester(2);
    }
    
    @Override
    public int[][] getInputDims() {
      return new int[][]{
        {3, 3, 2}
      };
    }
    
  }
//
//  @Override
//  public DerivativeTester getDerivativeTester() {
//    return new DerivativeTester(1e-1, 1e-2);
//  }
}
