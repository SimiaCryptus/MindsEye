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

package com.simiacryptus.mindseye.labs.encoding;

import com.simiacryptus.mindseye.eval.ArrayTrainable;
import com.simiacryptus.mindseye.eval.SampledArrayTrainable;
import com.simiacryptus.mindseye.eval.SampledTrainable;
import com.simiacryptus.mindseye.eval.TrainableDataMask;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.cudnn.ImgBandBiasLayer;
import com.simiacryptus.mindseye.layers.java.*;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.opt.Step;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.ValidatingTrainer;
import com.simiacryptus.mindseye.opt.line.QuadraticSearch;
import com.simiacryptus.mindseye.opt.orient.QQN;
import com.simiacryptus.util.FastRandom;
import com.simiacryptus.util.io.NotebookOutput;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;

/**
 * The type Find nn features.
 */
public class FindNNFeatures extends FindFeatureSpace {
  
  /**
   * The Space training minutes.
   */
  final int spaceTrainingMinutes;
  
  /**
   * Instantiates a new Find nn features.
   *
   * @param log                  the log
   * @param inputBands           the input bands
   * @param features             the features
   * @param spaceTrainingMinutes the space training minutes
   */
  public FindNNFeatures(NotebookOutput log, int inputBands, Tensor[][] features, int spaceTrainingMinutes) {
    super(log, inputBands, features);
    this.spaceTrainingMinutes = spaceTrainingMinutes;
  }
  
  @Override
  public FindFeatureSpace invoke() {
    ArrayList<Step> history = new ArrayList<>();
    TrainingMonitor monitor = TestUtil.getMonitor(history);
    int[] featureDimensions = features[0][1].getDimensions();
    int[] categoryDimensions = features[0][0].getDimensions();
    FullyConnectedLayer decodeMatrix = (FullyConnectedLayer) new FullyConnectedLayer(new int[]{inputBands}, featureDimensions).setWeights(() -> 0.001 * (FastRandom.random() - 0.5))
      .setName("renderMatrix");
    ImgBandBiasLayer bandBiasLayer = new ImgBandBiasLayer(featureDimensions[2]);
    
    PipelineNetwork network = log.code(() -> {
      PipelineNetwork pipelineNetwork = new PipelineNetwork(3);
      pipelineNetwork.add(decodeMatrix);
      pipelineNetwork.add(bandBiasLayer);
      DAGNode sqLoss = pipelineNetwork.add(new MeanSqLossLayer(), pipelineNetwork.getHead(), pipelineNetwork.getInput(1));
      
      pipelineNetwork.add(new FullyConnectedLayer(new int[]{inputBands}, categoryDimensions)
        .setWeights(() -> 0.00001 * (FastRandom.random() - 0.5))
        .setName("categoryMatrix"), pipelineNetwork.getInput(0));
      DAGNode predictionVector = pipelineNetwork.add(new BiasLayer(categoryDimensions));
      pipelineNetwork.add(new SoftmaxActivationLayer());
      DAGNode entropy = pipelineNetwork.add(new EntropyLossLayer(), pipelineNetwork.getHead(), pipelineNetwork.getInput(2));
      
      pipelineNetwork.add(new SumInputsLayer(),
        pipelineNetwork.add(new LinearActivationLayer().setScale(0.1).freeze(),
          pipelineNetwork.add(new SumReducerLayer(),
            pipelineNetwork.add(new AbsActivationLayer(), pipelineNetwork.getInput(0)))),
        pipelineNetwork.add(new LinearActivationLayer().setScale(0.001).freeze(),
          pipelineNetwork.add(new SumReducerLayer(),
            pipelineNetwork.add(new AbsActivationLayer(), predictionVector))),
        pipelineNetwork.add(new LinearActivationLayer().freeze(), sqLoss),
        pipelineNetwork.add(new LinearActivationLayer().freeze(), entropy));
      return pipelineNetwork;
    });
    
    TestUtil.addPerformanceWrappers(log, network);
    log.p(String.format("Training feature network for feature dims %s, category dims %s", Arrays.toString(featureDimensions), Arrays.toString(categoryDimensions)));
    Tensor[][] trainingData = log.code(() -> {
      return Arrays.stream(features).map(tensor -> new Tensor[]{
        new Tensor(inputBands).fill(() -> 0.01 * (FastRandom.random() - 0.5)), tensor[1], tensor[0]
      }).toArray(i -> new Tensor[i][]);
    });
    log.code(() -> {
      SampledTrainable trainingSubject = new SampledArrayTrainable(trainingData, network, trainingData.length / 1, trainingData.length);
      trainingSubject = (SampledTrainable) ((TrainableDataMask) trainingSubject).setMask(true, false, false);
      ValidatingTrainer validatingTrainer = new ValidatingTrainer(trainingSubject, new ArrayTrainable(trainingData, network))
        .setMaxTrainingSize(trainingData.length)
        .setMinTrainingSize(1)
        .setMonitor(monitor)
        .setTimeout(spaceTrainingMinutes, TimeUnit.MINUTES)
        .setMaxIterations(1000);
      validatingTrainer.getRegimen().get(0)
        .setOrientation(new QQN())
        .setLineSearchFactory(name -> {
          if (name.contains("LBFGS") || name.contains("QQN")) {
            return new QuadraticSearch().setCurrentRate(1.0);
            //return new ArmijoWolfeSearch().setAlpha(1.0).setMaxAlpha(1e8);
          }
          else {
            return new QuadraticSearch().setCurrentRate(1.0);
            //return new ArmijoWolfeSearch().setMaxAlpha(1e6);
          }
        });

//          validatingTrainer.getRegimen().get(0)
//            .setOrientation(new TrustRegionStrategy(new LBFGS()) {
//              @Override
//              public TrustRegion getRegionPolicy(NNLayer layer) {
//                if(layer instanceof BiasLayer) return null;
//                if(layer instanceof ImgBandBiasLayer) return null;
//                return new StaticConstraint();
//              }
//            })
//            .setLineSearchFactory(name -> {
//              if (name.contains("LBFGS") || name.contains("QQN")) {
//                return new QuadraticSearch().setCurrentRate(1.0);
//                //return new ArmijoWolfeSearch().setAlpha(1.0).setMaxAlpha(1e8);
//              }
//              else {
//                return new QuadraticSearch().setCurrentRate(1.0);
//                //return new ArmijoWolfeSearch().setMaxAlpha(1e6);
//              }
//            });
//          validatingTrainer.getRegimen().add(new ValidatingTrainer.TrainingPhase(trainingSubject)
//            .setOrientation(new TrustRegionStrategy(new LBFGS()) {
//              @Override
//              public TrustRegion getRegionPolicy(NNLayer layer) {
//                if(layer instanceof PlaceholderLayer) return null;
//                return new StaticConstraint();
//              }
//            })
//            .setLineSearchFactory(name -> {
//              if (name.contains("LBFGS") || name.contains("QQN")) {
//                return new QuadraticSearch().setCurrentRate(1.0);
//                //return new ArmijoWolfeSearch().setAlpha(1.0).setMaxAlpha(1e8);
//              }
//              else {
//                return new QuadraticSearch().setCurrentRate(1.0);
//                //return new ArmijoWolfeSearch().setMaxAlpha(1e6);
//              }
//            }));
//          validatingTrainer.getRegimen().add(new ValidatingTrainer.TrainingPhase(trainingSubject)
//            .setOrientation(new TrustRegionStrategy(new LBFGS()) {
//              @Override
//              public TrustRegion getRegionPolicy(NNLayer layer) {
//                if(layer instanceof FullyConnectedLayer) return null;
//                return new StaticConstraint();
//              }
//            })
//            .setLineSearchFactory(name -> {
//              if (name.contains("LBFGS") || name.contains("QQN")) {
//                return new QuadraticSearch().setCurrentRate(1.0);
//                //return new ArmijoWolfeSearch().setAlpha(1.0).setMaxAlpha(1e8);
//              }
//              else {
//                return new QuadraticSearch().setCurrentRate(1.0);
//                //return new ArmijoWolfeSearch().setMaxAlpha(1e6);
//              }
//            }));
      validatingTrainer
        .run();
      
      averages = Arrays.copyOf(bandBiasLayer.getBias(), bandBiasLayer.getBias().length);
      vectors = IntStream.range(0, inputBands).mapToObj(inputBand -> {
        Tensor to = new Tensor(featureDimensions);
        to.fillByCoord(c -> decodeMatrix.getWeights().get(c.index, inputBand));
        return to;
      }).toArray(i -> new Tensor[i]);
    });
    TestUtil.removePerformanceWrappers(log, network);
    TestUtil.printHistory(log, history);
    return this;
  }
}
