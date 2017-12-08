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

import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.orient.OrientationStrategy;
import com.simiacryptus.util.StreamNanoHTTPD;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.io.HtmlNotebookOutput;
import com.simiacryptus.util.io.NotebookOutput;

import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;

/**
 * The type Image encoding nn run.
 */
public class ImageEncodingNNTest extends ImageEncodingPCATest {
  
  /**
   * The Pretrain minutes.
   */
  int pretrainMinutes = 1;
  /**
   * The Timeout minutes.
   */
  int timeoutMinutes = 1;
  /**
   * The Space training minutes.
   */
  int spaceTrainingMinutes = 1;
  /**
   * The Size.
   */
  int size = 256;
  
  @Override
  public void run(NotebookOutput log) {
    Tensor[][] trainingImages = TestUtil.getImages(log, size, 30, "kangaroo", "yin_yang");
    
    log.h1("First Layer");
    InitializationStep step0 = log.code(() -> {
      return new InitializationStep(log, trainingImages,
        size, pretrainMinutes, timeoutMinutes, 3, 11, 5);
    }).invoke(); // 252
    
    log.h1("Second Layer");
    AddLayerStep step1 = log.code(() -> {
      return new AddLayerStep(log, step0.trainingData, step0.model,
        2, step0.toSize, pretrainMinutes, timeoutMinutes,
        step0.band1, 17, 7, 4);
    }).invoke(); // 224
    
    log.h1("Third Layer");
    AddLayerStep step2 = log.code(() -> {
      return new AddLayerStep(log, step1.trainingData, step1.integrationModel,
        3, step1.toSize, pretrainMinutes, timeoutMinutes,
        step1.band2, 13, 5, 2);
    }).invoke();
  }
  
  @Override
  public HtmlNotebookOutput report() {
    try {
      String directoryName = new SimpleDateFormat("YYYY-MM-dd-HH-mm").format(new Date());
      File path = new File(Util.mkString(File.separator, "www", directoryName));
      path.mkdirs();
      File logFile = new File(path, "index.html");
      StreamNanoHTTPD server = new StreamNanoHTTPD(1999, "text/html", logFile).init();
      HtmlNotebookOutput log = new HtmlNotebookOutput(path, server.dataReciever);
      log.addCopy(TestUtil.rawOut);
      return log;
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }
  
  @Override
  protected FindFeatureSpace findFeatureSpace(NotebookOutput log, Tensor[][] features, int inputBands) {
    return new FindNNFeatures(log, inputBands, features, spaceTrainingMinutes).invoke();
  }
  
  @Override
  protected void train(NotebookOutput log, TrainingMonitor monitor, NNLayer network, Tensor[][] data, OrientationStrategy orientation, int timeoutMinutes, boolean... mask) {
    if (network instanceof DAGNetwork) TestUtil.addPerformanceWrappers(log, (DAGNetwork) network);
    super.train(log, monitor, network, data, orientation, timeoutMinutes, mask);
    if (network instanceof DAGNetwork) TestUtil.removePerformanceWrappers(log, (DAGNetwork) network);
  }
  
}
