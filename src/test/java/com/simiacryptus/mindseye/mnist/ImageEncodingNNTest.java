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

package com.simiacryptus.mindseye.mnist;

import com.simiacryptus.mindseye.eval.*;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.activation.LinearActivationLayer;
import com.simiacryptus.mindseye.layers.activation.SoftmaxActivationLayer;
import com.simiacryptus.mindseye.layers.cudnn.f64.ImgBandBiasLayer;
import com.simiacryptus.mindseye.layers.loss.EntropyLossLayer;
import com.simiacryptus.mindseye.layers.loss.MeanSqLossLayer;
import com.simiacryptus.mindseye.layers.reducers.SumInputsLayer;
import com.simiacryptus.mindseye.layers.synapse.BiasLayer;
import com.simiacryptus.mindseye.layers.synapse.DenseSynapseLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.network.graph.DAGNode;
import com.simiacryptus.mindseye.opt.Step;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.ValidatingTrainer;
import com.simiacryptus.mindseye.opt.line.ArmijoWolfeSearch;
import com.simiacryptus.mindseye.opt.orient.QQN;
import com.simiacryptus.util.io.MarkdownNotebookOutput;
import com.simiacryptus.util.io.NotebookOutput;
import com.simiacryptus.util.test.TestCategories;
import org.junit.Test;
import org.junit.experimental.categories.Category;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;

public class ImageEncodingNNTest extends ImageEncodingPCATest {

  int pretrainMinutes = 1;
  int timeoutMinutes = 1;
  int spaceTrainingMinutes = 1;
  int size = 256;

  @Override
  @Test
  @Category(TestCategories.Report.class)
  public void test() throws Exception {
    try (NotebookOutput log = MarkdownNotebookOutput.get(this)) {
      if (null != out) ((MarkdownNotebookOutput) log).addCopy(out);
      
      
      Tensor[][] trainingImages = getImages(log, size, 10, "kangaroo", "yin_yang");
      
      log.h1("First Layer");
      InitializationStep step0 = log.code(()->{
        return new InitializationStep(log, trainingImages,
          size, pretrainMinutes, timeoutMinutes, 3, 7, 5);
      }).invoke();
  
      log.h1("Second Layer");
      AddLayerStep step1 = log.code(()->{
        return new AddLayerStep(log, step0.trainingData, step0.model,
          2, step0.toSize, pretrainMinutes, timeoutMinutes,
          step0.band1, 11, 5, 2);
      }).invoke();
  
    }
  }
  
  @Override
  protected FindFeatureSpace findFeatureSpace(NotebookOutput log, Tensor[][] features, int inputBands) {
    return new FindFeatureSpace(log, features, inputBands){
      @Override
      public FindFeatureSpace invoke() {
        ArrayList<Step> history = new ArrayList<>();
        TrainingMonitor monitor = getMonitor(out, history);
        log.code(() -> {
          int[] featureDimensions = features[0][1].getDimensions();
          PipelineNetwork network = new PipelineNetwork(3);
          DenseSynapseLayer synapseLayer = new DenseSynapseLayer(new int[]{inputBands}, featureDimensions);
          network.add(synapseLayer);
          ImgBandBiasLayer bandBiasLayer = new ImgBandBiasLayer(featureDimensions[2]);
          network.add(bandBiasLayer);
          DAGNode sqLoss = network.add(new MeanSqLossLayer(), network.getHead(), network.getInput(1));
  
          int[] categoryDimensions = features[0][0].getDimensions();
          network.add(new DenseSynapseLayer(new int[]{inputBands}, categoryDimensions), network.getInput(0));
          network.add(new BiasLayer(categoryDimensions));
          network.add(new SoftmaxActivationLayer());
          DAGNode entropy = network.add(new EntropyLossLayer(), network.getHead(), network.getInput(2));
          
          network.add(new SumInputsLayer(),
            network.add(new LinearActivationLayer().freeze(), sqLoss),
            network.add(new LinearActivationLayer().freeze(), entropy));
          
          Tensor[][] trainingData = Arrays.stream(features).map(tensor -> new Tensor[]{
            new Tensor(inputBands), tensor[1], tensor[0]
          }).toArray(i -> new Tensor[i][]);
    
          StochasticTrainable trainingSubject = new StochasticArrayTrainable(trainingData, network, trainingData.length);
          trainingSubject = (StochasticTrainable) ((TrainableDataMask) trainingSubject).setMask(true, false, false);
          new ValidatingTrainer(trainingSubject, new ArrayTrainable(trainingData, network))
            .setMaxTrainingSize(trainingData.length)
            .setMinTrainingSize(1)
            .setMonitor(monitor)
            .setOrientation(new QQN())
            .setTimeout(spaceTrainingMinutes, TimeUnit.MINUTES)
            .setLineSearchFactory(name -> {
              if (name.contains("LBFGS") || name.contains("QQN")) {
                return new ArmijoWolfeSearch().setAlpha(1.0).setMaxAlpha(1e8);
              }
              else {
                return new ArmijoWolfeSearch().setMaxAlpha(1e6);
              }
            })
            .setMaxIterations(1000)
            .run();
    
          averages = Arrays.copyOf(bandBiasLayer.getBias(), bandBiasLayer.getBias().length);
          vectors = IntStream.range(0, inputBands).mapToObj(inputBand->{
            Tensor to = new Tensor(featureDimensions);
            to.fillByCoord(c->synapseLayer.getWeights().get(inputBand, c.index));
            return to;
          }).toArray(i->new Tensor[i]);
        });
        printHistory(log, history);
        return this;
      }
    }.invoke();
  }
  
}
