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

package com.simiacryptus.mindseye.test.integration;

import com.simiacryptus.mindseye.eval.ArrayTrainable;
import com.simiacryptus.mindseye.eval.SampledArrayTrainable;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.StochasticComponent;
import com.simiacryptus.mindseye.layers.java.DropoutNoiseLayer;
import com.simiacryptus.mindseye.layers.java.MeanSqLossLayer;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.opt.Step;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.ValidatingTrainer;
import com.simiacryptus.mindseye.test.StepRecord;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.util.TableOutput;
import com.simiacryptus.util.io.NotebookOutput;
import com.simiacryptus.util.test.LabeledObject;
import guru.nidi.graphviz.engine.Format;
import guru.nidi.graphviz.engine.Graphviz;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * The type Mnist apply base.
 */
@SuppressWarnings("FieldCanBeLocal")
public class AutoencodingProblem implements Problem {
  
  private static int modelNo = 0;
  
  private final int batchSize = 10000;
  private final ImageProblemData data;
  private final double dropout;
  private final int features;
  private final FwdNetworkFactory fwdFactory;
  @Nonnull
  private final List<StepRecord> history = new ArrayList<>();
  private final OptimizationStrategy optimizer;
  private final RevNetworkFactory revFactory;
  private int timeoutMinutes = 1;
  
  /**
   * Instantiates a new Autoencoding problem.
   *
   * @param fwdFactory the fwd factory
   * @param optimizer  the optimizer
   * @param revFactory the rev factory
   * @param data       the data
   * @param features   the features
   * @param dropout    the dropout
   */
  public AutoencodingProblem(final FwdNetworkFactory fwdFactory, final OptimizationStrategy optimizer, final RevNetworkFactory revFactory, final ImageProblemData data, final int features, final double dropout) {
    this.fwdFactory = fwdFactory;
    this.optimizer = optimizer;
    this.revFactory = revFactory;
    this.data = data;
    this.features = features;
    this.dropout = dropout;
  }
  
  @Nonnull
  @Override
  public List<StepRecord> getHistory() {
    return history;
  }
  
  /**
   * Gets timeout minutes.
   *
   * @return the timeout minutes
   */
  public int getTimeoutMinutes() {
    return timeoutMinutes;
  }
  
  /**
   * Sets timeout minutes.
   *
   * @param timeoutMinutes the timeout minutes
   * @return the timeout minutes
   */
  @Nonnull
  public AutoencodingProblem setTimeoutMinutes(final int timeoutMinutes) {
    this.timeoutMinutes = timeoutMinutes;
    return this;
  }
  
  /**
   * Get training data tensor [ ] [ ].
   *
   * @param log the log
   * @return the tensor [ ] [ ]
   */
  public Tensor[][] getTrainingData(final NotebookOutput log) {
    try {
      return data.trainingData().map(labeledObject -> {
        return new Tensor[]{labeledObject.data};
      }).toArray(i -> new Tensor[i][]);
    } catch (@Nonnull final IOException e) {
      throw new RuntimeException(e);
    }
  }
  
  /**
   * Parse int.
   *
   * @param label the label
   * @return the int
   */
  public int parse(@Nonnull final String label) {
    return Integer.parseInt(label.replaceAll("[^\\d]", ""));
  }
  
  @Nonnull
  @Override
  public AutoencodingProblem run(@Nonnull final NotebookOutput log) {
    
    @Nonnull final DAGNetwork fwdNetwork = fwdFactory.imageToVector(log, features);
    @Nonnull final DAGNetwork revNetwork = revFactory.vectorToImage(log, features);
    
    @Nonnull final PipelineNetwork echoNetwork = new PipelineNetwork(1);
    echoNetwork.add(fwdNetwork).freeRef();
    echoNetwork.add(revNetwork).freeRef();
    
    @Nonnull final PipelineNetwork supervisedNetwork = new PipelineNetwork(1);
    supervisedNetwork.add(fwdNetwork).freeRef();
    @Nonnull final DropoutNoiseLayer dropoutNoiseLayer = new DropoutNoiseLayer().setValue(dropout);
    supervisedNetwork.add(dropoutNoiseLayer).freeRef();
    supervisedNetwork.add(revNetwork).freeRef();
    supervisedNetwork.wrap(new MeanSqLossLayer(),
      supervisedNetwork.getHead(),
      supervisedNetwork.getInput(0)).freeRef();
    
    log.h3("Network Diagrams");
    log.eval(() -> {
      return Graphviz.fromGraph(TestUtil.toGraph(fwdNetwork))
        .height(400).width(600).render(Format.PNG).toImage();
    });
    log.eval(() -> {
      return Graphviz.fromGraph(TestUtil.toGraph(revNetwork))
        .height(400).width(600).render(Format.PNG).toImage();
    });
    log.eval(() -> {
      return Graphviz.fromGraph(TestUtil.toGraph(supervisedNetwork))
        .height(400).width(600).render(Format.PNG).toImage();
    });
    
    @Nonnull final TrainingMonitor monitor = new TrainingMonitor() {
      @Nonnull
      TrainingMonitor inner = TestUtil.getMonitor(history);
      
      @Override
      public void log(final String msg) {
        inner.log(msg);
      }
      
      @Override
      public void onStepComplete(final Step currentPoint) {
        dropoutNoiseLayer.shuffle(StochasticComponent.random.get().nextLong());
        inner.onStepComplete(currentPoint);
      }
    };
    
    final Tensor[][] trainingData = getTrainingData(log);
    
    //MonitoredObject monitoringRoot = new MonitoredObject();
    //TestUtil.addMonitoring(supervisedNetwork, monitoringRoot);
    
    log.h3("Training");
    TestUtil.instrumentPerformance(supervisedNetwork);
    @Nonnull final ValidatingTrainer trainer = optimizer.train(log,
      new SampledArrayTrainable(trainingData, supervisedNetwork, trainingData.length / 2, batchSize),
      new ArrayTrainable(trainingData, supervisedNetwork, batchSize), monitor);
    log.run(() -> {
      trainer.setTimeout(timeoutMinutes, TimeUnit.MINUTES).setMaxIterations(10000).run();
    });
    if (!history.isEmpty()) {
      log.eval(() -> {
        return TestUtil.plot(history);
      });
      log.eval(() -> {
        return TestUtil.plotTime(history);
      });
    }
    TestUtil.extractPerformance(log, supervisedNetwork);
    
    {
      @Nonnull final String modelName = "encoder_model" + AutoencodingProblem.modelNo++ + ".json";
      log.p("Saved model as " + log.file(fwdNetwork.getJson().toString(), modelName, modelName));
    }
    
    @Nonnull final String modelName = "decoder_model" + AutoencodingProblem.modelNo++ + ".json";
    log.p("Saved model as " + log.file(revNetwork.getJson().toString(), modelName, modelName));

//    log.h3("Metrics");
//    log.run(() -> {
//      return TestUtil.toFormattedJson(monitoringRoot.getMetrics());
//    });
    
    log.h3("Validation");
    
    log.p("Here are some re-encoded examples:");
    log.eval(() -> {
      @Nonnull final TableOutput table = new TableOutput();
      data.validationData().map(labeledObject -> {
        return toRow(log, labeledObject, echoNetwork.eval(labeledObject.data).getData().get(0).getData());
      }).filter(x -> null != x).limit(10).forEach(table::putRow);
      return table;
    });
    
    log.p("Some rendered unit vectors:");
    for (int featureNumber = 0; featureNumber < features; featureNumber++) {
      @Nonnull final Tensor input = new Tensor(features).set(featureNumber, 1);
      @Nullable final Tensor tensor = revNetwork.eval(input).getData().get(0);
      log.out(log.png(tensor.toImage(), ""));
    }
    return this;
  }
  
  /**
   * To row linked hash buildMap.
   *
   * @param log              the log
   * @param labeledObject    the labeled object
   * @param predictionSignal the prediction signal
   * @return the linked hash buildMap
   */
  @Nonnull
  public LinkedHashMap<CharSequence, Object> toRow(@Nonnull final NotebookOutput log, @Nonnull final LabeledObject<Tensor> labeledObject, final double[] predictionSignal) {
    @Nonnull final LinkedHashMap<CharSequence, Object> row = new LinkedHashMap<>();
    row.put("Image", log.png(labeledObject.data.toImage(), labeledObject.label));
    row.put("Echo", log.png(new Tensor(predictionSignal, labeledObject.data.getDimensions()).toImage(), labeledObject.label));
    return row;
  }
}
