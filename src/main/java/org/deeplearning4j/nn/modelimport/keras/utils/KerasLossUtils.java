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
package org.deeplearning4j.nn.modelimport.keras.utils;

import org.deeplearning4j.nn.modelimport.keras.config.KerasLayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * Utility functionality for keras loss functions
 *
 * @author Max Pumperla
 */
public class KerasLossUtils {
  /**
   * Map Keras to DL4J loss functions.
   *
   * @param kerasLoss String containing Keras loss function name
   * @return String containing DL4J loss function
   */
  public static LossFunctions.LossFunction mapLossFunction(String kerasLoss, KerasLayerConfiguration conf)
    throws UnsupportedKerasConfigurationException {
    LossFunctions.LossFunction dl4jLoss;
    if (kerasLoss.equals(conf.getKERAS_LOSS_MEAN_SQUARED_ERROR()) ||
      kerasLoss.equals(conf.getKERAS_LOSS_MSE())) {
      dl4jLoss = LossFunctions.LossFunction.SQUARED_LOSS;
    }
    else if (kerasLoss.equals(conf.getKERAS_LOSS_MEAN_ABSOLUTE_ERROR()) ||
      kerasLoss.equals(conf.getKERAS_LOSS_MAE())) {
      dl4jLoss = LossFunctions.LossFunction.MEAN_ABSOLUTE_ERROR;
    }
    else if (kerasLoss.equals(conf.getKERAS_LOSS_MEAN_ABSOLUTE_PERCENTAGE_ERROR()) ||
      kerasLoss.equals(conf.getKERAS_LOSS_MAPE())) {
      dl4jLoss = LossFunctions.LossFunction.MEAN_ABSOLUTE_PERCENTAGE_ERROR;
    }
    else if (kerasLoss.equals(conf.getKERAS_LOSS_MEAN_SQUARED_LOGARITHMIC_ERROR()) ||
      kerasLoss.equals(conf.getKERAS_LOSS_MSLE())) {
      dl4jLoss = LossFunctions.LossFunction.MEAN_SQUARED_LOGARITHMIC_ERROR;
    }
    else if (kerasLoss.equals(conf.getKERAS_LOSS_SQUARED_HINGE())) {
      dl4jLoss = LossFunctions.LossFunction.SQUARED_HINGE;
    }
    else if (kerasLoss.equals(conf.getKERAS_LOSS_HINGE())) {
      dl4jLoss = LossFunctions.LossFunction.HINGE;
    }
    else if (kerasLoss.equals(conf.getKERAS_LOSS_SPARSE_CATEGORICAL_CROSSENTROPY())) {
      throw new UnsupportedKerasConfigurationException("Loss function " + kerasLoss + " not supported yet.");
    }
    else if (kerasLoss.equals(conf.getKERAS_LOSS_BINARY_CROSSENTROPY())) {
      dl4jLoss = LossFunctions.LossFunction.XENT;
    }
    else if (kerasLoss.equals(conf.getKERAS_LOSS_CATEGORICAL_CROSSENTROPY())) {
      dl4jLoss = LossFunctions.LossFunction.MCXENT;
    }
    else if (kerasLoss.equals(conf.getKERAS_LOSS_KULLBACK_LEIBLER_DIVERGENCE()) ||
      kerasLoss.equals(conf.getKERAS_LOSS_KLD())) {
      dl4jLoss = LossFunctions.LossFunction.KL_DIVERGENCE;
    }
    else if (kerasLoss.equals(conf.getKERAS_LOSS_POISSON())) {
      dl4jLoss = LossFunctions.LossFunction.POISSON;
    }
    else if (kerasLoss.equals(conf.getKERAS_LOSS_COSINE_PROXIMITY())) {
      dl4jLoss = LossFunctions.LossFunction.COSINE_PROXIMITY;
    }
    else {
      throw new UnsupportedKerasConfigurationException("Unknown Keras loss function " + kerasLoss);
    }
    return dl4jLoss;
  }
}
