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

package com.simiacryptus.mindseye.demo;

import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.cudnn.CudaSystem;
import com.simiacryptus.mindseye.models.VGG16;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.test.NotebookReportBase;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.util.FastRandom;
import com.simiacryptus.util.StreamNanoHTTPD;
import com.simiacryptus.util.data.DoubleStatistics;
import com.simiacryptus.util.io.JsonUtil;
import com.simiacryptus.util.io.NotebookOutput;
import org.apache.hadoop.yarn.webapp.MimeType;

import javax.annotation.Nonnull;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Random;
import java.util.stream.IntStream;

/**
 * The type Artistry demo.
 */
public class ArtistryDemo extends NotebookReportBase {
  
  /**
   * The Server.
   */
  StreamNanoHTTPD server;
  
  /**
   * Add layers handler.
   *
   * @param painterNetwork the painter network
   * @param server         the server
   */
  public static void addLayersHandler(final DAGNetwork painterNetwork, final StreamNanoHTTPD server) {
    server.addSyncHandler("layers.json", MimeType.JSON, out -> {
      try {
        JsonUtil.MAPPER.writer().writeValue(out, TestUtil.samplePerformance(painterNetwork));
        out.close();
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    }, false);
  }
  
  /**
   * Paint low res.
   *
   * @param canvas the canvas
   * @param scale  the scale
   */
  public void paint_LowRes(final Tensor canvas, final int scale) {
    BufferedImage originalImage = canvas.toImage();
    canvas.set(Tensor.fromRGB(TestUtil.resize(
      TestUtil.resize(originalImage, originalImage.getWidth() / scale, true),
      originalImage.getWidth(), originalImage.getHeight())));
  }
  
  /**
   * Paint lines.
   *
   * @param canvas the canvas
   */
  public void paint_Lines(final Tensor canvas) {
    BufferedImage originalImage = canvas.toImage();
    BufferedImage newImage = new BufferedImage(originalImage.getWidth(), originalImage.getHeight(), BufferedImage.TYPE_INT_ARGB);
    Graphics2D graphics = (Graphics2D) newImage.getGraphics();
    IntStream.range(0, 100).forEach(i -> {
      Random random = new Random();
      graphics.setColor(new Color(random.nextInt(255), random.nextInt(255), random.nextInt(255)));
      graphics.drawLine(
        random.nextInt(originalImage.getWidth()),
        random.nextInt(originalImage.getHeight()),
        random.nextInt(originalImage.getWidth()),
        random.nextInt(originalImage.getHeight())
      );
    });
    canvas.set(Tensor.fromRGB(newImage));
  }
  
  /**
   * Paint circles.
   *
   * @param canvas the canvas
   * @param scale  the scale
   */
  public void paint_Circles(final Tensor canvas, final int scale) {
    BufferedImage originalImage = canvas.toImage();
    BufferedImage newImage = new BufferedImage(originalImage.getWidth(), originalImage.getHeight(), BufferedImage.TYPE_INT_ARGB);
    Graphics2D graphics = (Graphics2D) newImage.getGraphics();
    IntStream.range(0, 10000).forEach(i -> {
      Random random = new Random();
      int positionX = random.nextInt(originalImage.getWidth());
      int positionY = random.nextInt(originalImage.getHeight());
      int width = 1 + random.nextInt(2 * scale);
      int height = 1 + random.nextInt(2 * scale);
      DoubleStatistics[] stats = {
        new DoubleStatistics(),
        new DoubleStatistics(),
        new DoubleStatistics()
      };
      canvas.coordStream(false).filter(c -> {
        int[] coords = c.getCoords();
        int x = coords[0];
        int y = coords[1];
        double relX = Math.pow(1 - 2 * ((double) (x - positionX) / width), 2);
        double relY = Math.pow(1 - 2 * ((double) (y - positionY) / height), 2);
        return relX + relY < 1.0;
      }).forEach(c -> stats[c.getCoords()[2]].accept(canvas.get(c)));
      graphics.setStroke(new Stroke() {
        @Override
        public Shape createStrokedShape(final Shape p) {
          return null;
        }
      });
      graphics.setColor(new Color(
        (int) stats[0].getAverage(),
        (int) stats[1].getAverage(),
        (int) stats[2].getAverage()
      ));
      graphics.fillOval(
        positionX,
        positionY,
        width,
        height
      );
    });
    canvas.set(Tensor.fromRGB(newImage));
  }
  
  /**
   * Gets target class.
   *
   * @return the target class
   */
  @Nonnull
  protected Class<?> getTargetClass() {
    return VGG16.class;
  }
  
  @Nonnull
  @Override
  public ReportType getReportType() {
    return ReportType.Demos;
  }
  
  /**
   * Paint noise.
   *
   * @param canvas the canvas
   */
  public void paint_noise(final Tensor canvas) {
    canvas.setByCoord(c -> FastRandom.INSTANCE.random());
  }
  
  /**
   * Init.
   *
   * @param log the log
   */
  public void init(final NotebookOutput log) {
    try {
      server = new StreamNanoHTTPD(9090).init();
      server.addSyncHandler("gpu.json", MimeType.JSON, out -> {
        try {
          JsonUtil.MAPPER.writer().writeValue(out, CudaSystem.getExecutionStatistics());
          //JsonUtil.MAPPER.writer().writeValue(out, new HashMap<>());
          out.close();
        } catch (IOException e) {
          throw new RuntimeException(e);
        }
      }, false);
      //server.dataReciever
      //server.init();
      //server.start();
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
    @Nonnull String logName = "cuda_" + log.getName() + ".log";
    log.p(log.file((String) null, logName, "GPU Log"));
    CudaSystem.addLog(new PrintStream(log.file(logName)));
  }
}
