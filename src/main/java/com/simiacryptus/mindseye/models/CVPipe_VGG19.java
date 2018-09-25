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

import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.util.Util;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

/**
 * The type Multi layer vgg 16.
 */
public class CVPipe_VGG19 implements CVPipe<CVPipe_VGG19.Layer> {

  /**
   * The constant logger.
   */
  public static final Logger logger = LoggerFactory.getLogger(CVPipe_VGG19.class);
  /**
   * The constant INSTANCE.
   */
  public static final CVPipe_VGG19 INSTANCE = build();
  private final Map<Layer, UUID> nodes = new HashMap<>();
  private final Map<Layer, PipelineNetwork> prototypes = new HashMap<>();
  private PipelineNetwork network;

  private CVPipe_VGG19() {
  }

  private static CVPipe_VGG19 build() {
    CVPipe_VGG19 obj = new CVPipe_VGG19();
    final String abortMsg = "Abort Network Construction";
    try {
      new VGG19_HDF5(new Hdf5Archive(Util.cacheFile(TestUtil.S3_ROOT.resolve("vgg19_weights.h5")))) {
        @Override
        protected void phase0() {
          super.phase0();
          obj.nodes.put(Layer.Layer_0, pipeline.getHeadId());
          obj.prototypes.put(Layer.Layer_0, pipeline.copy());
        }

        @Override
        protected void phase1a() {
          super.phase1a();
          obj.nodes.put(Layer.Layer_1a, pipeline.getHeadId());
          obj.prototypes.put(Layer.Layer_1a, pipeline.copy());
        }

        @Override
        protected void phase1b() {
          super.phase1b();
          obj.nodes.put(Layer.Layer_1b, pipeline.getHeadId());
          obj.prototypes.put(Layer.Layer_1b, pipeline.copy());
        }

        @Override
        protected void phase1c() {
          super.phase1c();
          obj.nodes.put(Layer.Layer_1c, pipeline.getHeadId());
          obj.prototypes.put(Layer.Layer_1c, pipeline.copy());
        }

        @Override
        protected void phase1d() {
          super.phase1d();
          obj.nodes.put(Layer.Layer_1d, pipeline.getHeadId());
          obj.prototypes.put(Layer.Layer_1d, pipeline.copy());
        }

        @Override
        protected void phase1e() {
          super.phase1e();
          obj.nodes.put(Layer.Layer_1e, pipeline.getHeadId());
          obj.prototypes.put(Layer.Layer_1e, pipeline.copy());
        }

        @Override
        protected void phase2a() {
          super.phase2a();
          obj.nodes.put(Layer.Layer_2a, pipeline.getHeadId());
          obj.prototypes.put(Layer.Layer_2a, pipeline.copy());
        }

        @Override
        protected void phase2b() {
          super.phase2b();
          obj.nodes.put(Layer.Layer_2b, pipeline.getHeadId());
          obj.prototypes.put(Layer.Layer_2b, pipeline.copy());
        }

        @Override
        protected void phase3a() {
          super.phase3a();
          obj.nodes.put(Layer.Layer_3a, pipeline.getHeadId());
          obj.prototypes.put(Layer.Layer_3a, pipeline.copy());
        }

        @Override
        protected void phase3b() {
          super.phase3b();
          obj.nodes.put(Layer.Layer_3b, pipeline.getHeadId());
          obj.prototypes.put(Layer.Layer_3b, pipeline.copy());
          obj.network = (PipelineNetwork) pipeline.freeze();
          throw new RuntimeException(abortMsg);
        }
      }.getNetwork();
      assert null != obj.prototypes;
      assert !obj.prototypes.isEmpty();
    } catch (@Nonnull final RuntimeException e1) {
      if (!e1.getMessage().equals(abortMsg)) {
        logger.warn("Err", e1);
        throw new RuntimeException(e1);
      }
    } catch (Throwable e11) {
      logger.warn("Error", e11);
      throw new RuntimeException(e11);
    }
    return obj;
  }

  @Override
  public Map<Layer, UUID> getNodes() {
    return Collections.unmodifiableMap(nodes);
  }

  @Override
  public Map<Layer, PipelineNetwork> getPrototypes() {
    assert null != prototypes;
    assert !prototypes.isEmpty();
    return Collections.unmodifiableMap(prototypes);
  }

  @Override
  public PipelineNetwork getNetwork() {
    return network.copy();
  }

  /**
   * The enum Layer type.
   */
  public enum Layer implements LayerEnum<Layer> {
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
    Layer_1e, /**
     * Layer 2 a layer type.
     */
    Layer_2a, /**
     * Layer 2 b layer type.
     */
    Layer_2b, /**
     * Layer 3 a layer type.
     */
    Layer_3a, /**
     * Layer 3 b layer type.
     */
    Layer_3b;

    /**
     * Texture pipeline network.
     *
     * @return the pipeline network
     */
    public final PipelineNetwork network() {
      PipelineNetwork pipelineNetwork = INSTANCE.getPrototypes().get(this);
      if (null == pipelineNetwork) throw new IllegalStateException(this.toString());
      return null == pipelineNetwork ? null : pipelineNetwork.copy();
    }
  }
}
