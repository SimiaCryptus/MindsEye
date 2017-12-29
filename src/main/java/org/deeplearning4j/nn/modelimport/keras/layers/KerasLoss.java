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

package org.deeplearning4j.nn.modelimport.keras.layers;

import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.ArrayList;

/**
 * Builds a DL4J LossLayer from a Keras training loss function.
 *
 * @author dave@skymind.io
 */
public class KerasLoss extends KerasLayer {
  
  private final String KERAS_CLASS_NAME_LOSS = "Loss";
  
  /**
   * Constructor from layer name and input shape.
   *
   * @param layerName        layer name
   * @param inboundLayerName name of inbound layer
   * @param kerasLoss        name of Keras loss function
   * @throws InvalidKerasConfigurationException
   * @throws UnsupportedKerasConfigurationException
   */
  public KerasLoss(String layerName, String inboundLayerName, String kerasLoss)
    throws UnsupportedKerasConfigurationException {
    this(layerName, inboundLayerName, kerasLoss, true);
  }
  
  /**
   * Constructor from layer name and input shape.
   *
   * @param layerName             layer name
   * @param inboundLayerName      name of inbound layer
   * @param kerasLoss             name of Keras loss function
   * @param enforceTrainingConfig whether to enforce training-related configuration options
   * @throws InvalidKerasConfigurationException
   * @throws UnsupportedKerasConfigurationException
   */
  public KerasLoss(String layerName, String inboundLayerName, String kerasLoss, boolean enforceTrainingConfig)
    throws UnsupportedKerasConfigurationException {
    this.className = KERAS_CLASS_NAME_LOSS;
    this.layerName = layerName;
    this.inputShape = null;
    this.dimOrder = DimOrder.NONE;
    this.inboundLayerNames = new ArrayList<String>();
    this.inboundLayerNames.add(inboundLayerName);
    LossFunctions.LossFunction loss;
    throw new RuntimeException("NI");
//        try {
//            loss = mapLossFunction(kerasLoss, conf);
//        } catch (UnsupportedKerasConfigurationException e) {
//            if (enforceTrainingConfig)
//                throw e;
//            log.warn("Unsupported Keras loss function. Replacing with MSE.");
//            loss = LossFunctions.LossFunction.SQUARED_LOSS;
//        }
//        this.layer = new LossLayer.Builder(loss).name(layerName).build();
  }

//    /**
//     * Get DL4J LossLayer.
//     *
//     * @return  LossLayer
//     */
//    public LossLayer getLossLayer() {
//        return (LossLayer) this.layer;
//    }
//
  
  /**
   * Get layer output type.
   *
   * @param inputType Array of InputTypes
   * @return output type as InputType
   * @throws InvalidKerasConfigurationException
   */
  @Override
  public InputType getOutputType(InputType... inputType) throws InvalidKerasConfigurationException {
      if (inputType.length > 1) {
          throw new InvalidKerasConfigurationException(
            "Keras Loss layer accepts only one input (received " + inputType.length + ")");
      }
    throw new RuntimeException("NI");
//        return this.getLossLayer().getOutputType(-1, inputType[0]);
  }
}
