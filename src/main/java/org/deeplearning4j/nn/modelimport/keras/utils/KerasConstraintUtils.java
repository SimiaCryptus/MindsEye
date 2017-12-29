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


/**
 * Utility functionality for keras constraints.
 *
 * @author Max Pumperla
 */
public class KerasConstraintUtils {
//
//    /**
//     * Map Keras to DL4J constraint.
//     *
//     * @param kerasConstraint String containing Keras constraint name
//     * @param conf            Keras layer configuration
//     * @return DL4J LayerConstraint
//     * @see LayerConstraint
//     */
//    public static LayerConstraint mapConstraint(String kerasConstraint, KerasLayerConfiguration conf,
//                                                Map<String, Object> constraintConfig)
//            throws UnsupportedKerasConfigurationException {
//        LayerConstraint constraint;
//        if (kerasConstraint.equals(conf.getLAYER_FIELD_MINMAX_NORM_CONSTRAINT())
//                || kerasConstraint.equals(conf.getLAYER_FIELD_MINMAX_NORM_CONSTRAINT_ALIAS())) {
//            double min = (double) constraintConfig.get(conf.getLAYER_FIELD_MINMAX_MIN_CONSTRAINT());
//            double max = (double) constraintConfig.get(conf.getLAYER_FIELD_MINMAX_MAX_CONSTRAINT());
//            double rate = (double) constraintConfig.get(conf.getLAYER_FIELD_CONSTRAINT_RATE());
//            int dim = (int) constraintConfig.get(conf.getLAYER_FIELD_CONSTRAINT_DIM());
//            constraint = new MinMaxNormConstraint(min, max, rate, dim + 1);
//        } else if (kerasConstraint.equals(conf.getLAYER_FIELD_MAX_NORM_CONSTRAINT())
//                || kerasConstraint.equals(conf.getLAYER_FIELD_MAX_NORM_CONSTRAINT_ALIAS())
//                || kerasConstraint.equals(conf.getLAYER_FIELD_MAX_NORM_CONSTRAINT_ALIAS_2())) {
//            double max = (double) constraintConfig.get(conf.getLAYER_FIELD_MAX_CONSTRAINT());
//            int dim = (int) constraintConfig.get(conf.getLAYER_FIELD_CONSTRAINT_DIM());
//            constraint = new MaxNormConstraint(max, dim + 1);
//        } else if (kerasConstraint.equals(conf.getLAYER_FIELD_UNIT_NORM_CONSTRAINT())
//                || kerasConstraint.equals(conf.getLAYER_FIELD_UNIT_NORM_CONSTRAINT_ALIAS())
//                || kerasConstraint.equals(conf.getLAYER_FIELD_UNIT_NORM_CONSTRAINT_ALIAS_2())) {
//            int dim = (int) constraintConfig.get(conf.getLAYER_FIELD_CONSTRAINT_DIM());
//            constraint = new UnitNormConstraint(dim + 1);
//        } else if (kerasConstraint.equals(conf.getLAYER_FIELD_NON_NEG_CONSTRAINT())
//                || kerasConstraint.equals(conf.getLAYER_FIELD_NON_NEG_CONSTRAINT_ALIAS())
//                || kerasConstraint.equals(conf.getLAYER_FIELD_NON_NEG_CONSTRAINT_ALIAS_2())) {
//            constraint = new NonNegativeConstraint();
//        }
//        else {
//            throw new UnsupportedKerasConfigurationException("Unknown keras constraint " + kerasConstraint);
//        }
//
//        return constraint;
//    }
//
//    /**
//     * Get constraint initialization from Keras layer configuration.
//     *
//     * @param layerConfig       dictionary containing Keras layer configuration
//     * @param constraintField   string in configuration representing parameter to constrain
//     * @param conf              Keras layer configuration
//     * @param kerasMajorVersion Major keras version as integer (1 or 2)
//     * @return a valid LayerConstraint
//     * @throws InvalidKerasConfigurationException     Invalid configuration
//     * @throws UnsupportedKerasConfigurationException Unsupported configuration
//     */
//    public static LayerConstraint getConstraintsFromConfig(Map<String, Object> layerConfig, String constraintField,
//                                                           KerasLayerConfiguration conf, int kerasMajorVersion)
//            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
//        Map<String, Object> innerConfig = KerasLayerUtils.getInnerLayerConfigFromConfig(layerConfig, conf);
//        if (!innerConfig.containsKey(constraintField)) {
//            // log.warn("Keras layer is missing " + constraintField + " field");
//            return null;
//        }
//        HashMap  constraintMap = (HashMap) innerConfig.get(constraintField);
//        if (constraintMap == null)
//            return null;
//
//        String kerasConstraint;
//        if (constraintMap.containsKey(conf.getLAYER_FIELD_CONSTRAINT_NAME())) {
//            kerasConstraint = (String) constraintMap.get(conf.getLAYER_FIELD_CONSTRAINT_NAME());
//        } else {
//            throw new InvalidKerasConfigurationException("Keras layer is missing " +
//                    conf.getLAYER_FIELD_CONSTRAINT_NAME() + " field");
//        }
//
//        Map<String, Object> constraintConfig;
//        if (kerasMajorVersion == 2) {
//            constraintConfig = KerasLayerUtils.getInnerLayerConfigFromConfig(constraintMap, conf);
//        } else {
//            constraintConfig = constraintMap;
//        }
//        LayerConstraint layerConstraint = mapConstraint(kerasConstraint, conf, constraintConfig);
//
//        return layerConstraint;
//    }
}
