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

package com.simiacryptus.mindseye.models;

import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.util.Util;

import javax.annotation.Nonnull;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

/**
 * The type Multi layer vgg 16.
 */
public class MultiLayerVGG16 implements MultiLayerImageNetwork<MultiLayerVGG16.LayerType> {
  /**
   * The constant INSTANCE.
   */
  public static final MultiLayerImageNetwork<LayerType> INSTANCE = build();
  private final Map<LayerType, UUID> nodes = new HashMap<>();
  private final Map<LayerType, PipelineNetwork> prototypes = new HashMap<>();
  private PipelineNetwork network = new PipelineNetwork();
  
  private static MultiLayerImageNetwork<LayerType> build() {
    MultiLayerVGG16 obj = new MultiLayerVGG16();
    final DAGNode[] nodes = new DAGNode[6];
    try {
      new VGG16_HDF5(new Hdf5Archive(Util.cacheFile(TestUtil.S3_ROOT.resolve("vgg16_weights.h5")))) {
        @Override
        protected void phase0() {
          super.phase0();
          nodes[0] = pipeline.getHead();
          obj.nodes.put(LayerType.Layer_0, pipeline.getHead().getId());
          obj.prototypes.put(LayerType.Layer_0, pipeline.copy());
        }
        
        @Override
        protected void phase1a() {
          super.phase1a();
          nodes[1] = pipeline.getHead();
          obj.nodes.put(LayerType.Layer_1a, pipeline.getHead().getId());
          obj.prototypes.put(LayerType.Layer_1a, pipeline.copy());
        }
        
        @Override
        protected void phase1b() {
          super.phase1b();
          nodes[2] = pipeline.getHead();
          obj.nodes.put(LayerType.Layer_1b, pipeline.getHead().getId());
          obj.prototypes.put(LayerType.Layer_1b, pipeline.copy());
        }
        
        @Override
        protected void phase1c() {
          super.phase1c();
          nodes[3] = pipeline.getHead();
          obj.nodes.put(LayerType.Layer_1c, pipeline.getHead().getId());
          obj.prototypes.put(LayerType.Layer_1c, pipeline.copy());
        }
        
        @Override
        protected void phase1d() {
          super.phase1d();
          nodes[4] = pipeline.getHead();
          obj.nodes.put(LayerType.Layer_1d, pipeline.getHead().getId());
          obj.prototypes.put(LayerType.Layer_1d, pipeline.copy());
        }
        
        @Override
        protected void phase1e() {
          super.phase1e();
          nodes[5] = pipeline.getHead();
          obj.nodes.put(LayerType.Layer_1e, pipeline.getHead().getId());
          obj.prototypes.put(LayerType.Layer_1e, pipeline.copy());
          obj.network = (PipelineNetwork) pipeline.freeze();
          throw new RuntimeException("Abort Network Construction");
        }
      }.getNetwork();
    } catch (@Nonnull final RuntimeException e1) {
    } catch (Throwable e11) {
      throw new RuntimeException(e11);
    }
    return obj;
  }
  
  @Override
  public Map<LayerType, UUID> getNodes() {
    return Collections.unmodifiableMap(nodes);
  }
  
  @Override
  public Map<LayerType, PipelineNetwork> getPrototypes() {
    return Collections.unmodifiableMap(prototypes);
  }
  
  @Override
  public PipelineNetwork getNetwork() {
    return network.copy();
  }
  
  /**
   * The enum Layer type.
   */
  public enum LayerType {
    /**
     * Layer 0 layer type.
     */
    Layer_0,
    /**
     * Layer 1 a layer type.
     */
    Layer_1a,
    /**
     * Layer 1 b layer type.
     */
    Layer_1b,
    /**
     * Layer 1 c layer type.
     */
    Layer_1c,
    /**
     * Layer 1 d layer type.
     */
    Layer_1d,
    /**
     * Layer 1 e layer type.
     */
    Layer_1e;
    
    /**
     * Texture pipeline network.
     *
     * @return the pipeline network
     */
    public final PipelineNetwork texture() {
      PipelineNetwork pipelineNetwork = INSTANCE.getPrototypes().get(this);
      if (null == pipelineNetwork) throw new IllegalStateException(this.toString());
      return null == pipelineNetwork ? null : pipelineNetwork.copy();
    }
  }
}
