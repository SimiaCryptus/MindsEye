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

package com.simiacryptus.mindseye.applications;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.layers.cudnn.ActivationLayer;
import com.simiacryptus.mindseye.layers.cudnn.BandAvgReducerLayer;
import com.simiacryptus.mindseye.layers.cudnn.BandReducerLayer;
import com.simiacryptus.mindseye.layers.cudnn.ConvolutionLayer;
import com.simiacryptus.mindseye.layers.cudnn.GramianLayer;
import com.simiacryptus.mindseye.layers.cudnn.ImgBandBiasLayer;
import com.simiacryptus.mindseye.layers.cudnn.MultiPrecision;
import com.simiacryptus.mindseye.layers.cudnn.PoolingLayer;
import com.simiacryptus.mindseye.layers.cudnn.SquareActivationLayer;
import com.simiacryptus.mindseye.layers.java.LinearActivationLayer;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.test.PCAUtil;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.util.FastRandom;
import com.simiacryptus.util.StreamNanoHTTPD;
import com.simiacryptus.util.data.DoubleStatistics;
import com.simiacryptus.util.io.JsonUtil;
import com.simiacryptus.util.io.NotebookOutput;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.hadoop.yarn.webapp.MimeType;

import javax.annotation.Nonnull;
import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Random;
import java.util.function.DoubleUnaryOperator;
import java.util.function.Supplier;
import java.util.stream.IntStream;

/**
 * The type Artistry util.
 */
public class ArtistryUtil {
  /**
   * Add layers handler.
   *
   * @param painterNetwork the painter network
   * @param server         the server
   */
  public static void addLayersHandler(final DAGNetwork painterNetwork, final StreamNanoHTTPD server) {
    if (null != server) server.addSyncHandler("layers.json", MimeType.JSON, out -> {
      try {
        JsonUtil.MAPPER.writer().writeValue(out, TestUtil.samplePerformance(painterNetwork));
        out.close();
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    }, false);
  }
  
  /**
   * Gram pipeline network.
   *
   * @param network      the network
   * @param mean         the mean
   * @param pcaTransform the pca transform
   * @return the pipeline network
   */
  @Nonnull
  public static PipelineNetwork gram(final PipelineNetwork network, Tensor mean, Tensor pcaTransform) {
    int[] dimensions = pcaTransform.getDimensions();
    int inputBands = mean.getDimensions()[2];
    int pcaBands = dimensions[2];
    int outputBands = pcaBands / inputBands;
    int width = dimensions[0];
    int height = dimensions[1];
    network.wrap(new ImgBandBiasLayer(mean.scale(-1)));
    network.wrap(new ConvolutionLayer(width, height, inputBands, outputBands).set(pcaTransform));
    network.wrap(new GramianLayer());
    return network;
  }
  
  /**
   * Square avg pipeline network.
   *
   * @param network      the network
   * @param mean         the mean
   * @param pcaTransform the pca transform
   * @return the pipeline network
   */
  @Nonnull
  public static PipelineNetwork squareAvg(final PipelineNetwork network, Tensor mean, Tensor pcaTransform) {
    int[] dimensions = pcaTransform.getDimensions();
    int inputBands = mean.getDimensions()[2];
    int pcaBands = dimensions[2];
    int outputBands = pcaBands / inputBands;
    int width = dimensions[0];
    int height = dimensions[1];
    network.wrap(new ImgBandBiasLayer(mean.scale(-1)));
    network.wrap(new ConvolutionLayer(width, height, inputBands, outputBands).set(pcaTransform));
    network.wrap(new SquareActivationLayer());
    network.wrap(new BandAvgReducerLayer());
    return network;
  }
  
  /**
   * Paint low res.
   *
   * @param canvas the canvas
   * @param scale  the scale
   */
  public static void paint_LowRes(final Tensor canvas, final int scale) {
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
  public static void paint_Lines(final Tensor canvas) {
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
  public static void paint_Circles(final Tensor canvas, final int scale) {
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
   * Avg pipeline network.
   *
   * @param network the network
   * @return the pipeline network
   */
  @Nonnull
  public static PipelineNetwork avg(final PipelineNetwork network) {
    network.wrap(new BandReducerLayer().setMode(PoolingLayer.PoolingMode.Avg));
    return network;
  }
  
  /**
   * With clamp pipeline network.
   *
   * @param network1 the network 1
   * @return the pipeline network
   */
  @Nonnull
  public static PipelineNetwork withClamp(final PipelineNetwork network1) {
    PipelineNetwork network = new PipelineNetwork(1);
    network.wrap(getClamp(255));
    network.wrap(network1);
    return network;
  }
  
  /**
   * Sets precision.
   *
   * @param network   the network
   * @param precision the precision
   */
  public static void setPrecision(final DAGNetwork network, final Precision precision) {
    network.visitLayers(layer -> {
      if (layer instanceof MultiPrecision) {
        ((MultiPrecision) layer).setPrecision(precision);
      }
    });
  }
  
  /**
   * Pca tensor.
   *
   * @param cov   the cov
   * @param power the power
   * @return the tensor
   */
  @Nonnull
  public static Tensor pca(final Tensor cov, final double power) {
    final int inputbands = (int) Math.sqrt(cov.getDimensions()[2]);
    final int outputbands = inputbands;
    Array2DRowRealMatrix realMatrix = new Array2DRowRealMatrix(inputbands, inputbands);
    cov.coordStream(false).forEach(c -> {
      double v = cov.get(c);
      int x = c.getIndex() % inputbands;
      int y = (c.getIndex() - x) / inputbands;
      realMatrix.setEntry(x, y, v);
    });
    Tensor[] features = PCAUtil.pcaFeatures(realMatrix, outputbands, new int[]{1, 1, inputbands}, power);
    Tensor kernel = new Tensor(1, 1, inputbands * outputbands);
    PCAUtil.populatePCAKernel_1(kernel, features);
    return kernel;
  }
  
  /**
   * Gets clamp.
   *
   * @param max the max
   * @return the clamp
   */
  @Nonnull
  public static PipelineNetwork getClamp(final int max) {
    @Nonnull PipelineNetwork clamp = new PipelineNetwork(1);
    clamp.add(new ActivationLayer(ActivationLayer.Mode.RELU));
    clamp.add(new LinearActivationLayer().setBias(max).setScale(-1).freeze());
    clamp.add(new ActivationLayer(ActivationLayer.Mode.RELU));
    clamp.add(new LinearActivationLayer().setBias(max).setScale(-1).freeze());
    return clamp;
  }
  
  /**
   * To json string.
   *
   * @param obj the style parameters
   * @return the string
   */
  public static CharSequence toJson(final Object obj) {
    String json;
    try {
      ObjectMapper mapper = new ObjectMapper();
      mapper.configure(SerializationFeature.INDENT_OUTPUT, true);
      json = mapper.writeValueAsString(obj);
    } catch (JsonProcessingException e) {
      throw new RuntimeException(e);
    }
    return json;
  }
  
  /**
   * Load buffered image.
   *
   * @param image the style
   * @return the buffered image
   */
  @Nonnull
  public static BufferedImage load(final CharSequence image) {
    BufferedImage bufferedImage;
    try {
      bufferedImage = ImageIO.read(new File(image.toString()));
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
    return bufferedImage;
  }
  
  /**
   * Load buffered image.
   *
   * @param image     the image
   * @param imageSize the image size
   * @return the buffered image
   */
  @Nonnull
  public static BufferedImage load(final String image, final int imageSize) {
    BufferedImage bufferedImage;
    try {
      bufferedImage = ImageIO.read(new File(image));
      bufferedImage = TestUtil.resize(bufferedImage, imageSize, true);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
    return bufferedImage;
  }
  
  /**
   * Load buffered image.
   *
   * @param imageFile the style
   * @param width     the width
   * @param height    the height
   * @return the buffered image
   */
  @Nonnull
  public static BufferedImage load(final String imageFile, final int width, final int height) {
    BufferedImage image;
    try {
      image = ImageIO.read(new File(imageFile));
      image = TestUtil.resize(image, width, height);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
    return image;
  }
  
  /**
   * Gram pipeline network.
   *
   * @param network the network
   * @param mean    the mean
   * @return the pipeline network
   */
  @Nonnull
  public static PipelineNetwork gram(final PipelineNetwork network, Tensor mean) {
    network.wrap(new ImgBandBiasLayer(mean.scale(-1)));
    network.wrap(new GramianLayer());
    return network;
  }
  
  /**
   * Gram pipeline network.
   *
   * @param network the network
   * @return the pipeline network
   */
  @Nonnull
  public static PipelineNetwork gram(final PipelineNetwork network) {
    network.wrap(new GramianLayer());
    return network;
  }
  
  /**
   * Randomize buffered image.
   *
   * @param contentImage the content image
   * @return the buffered image
   */
  @Nonnull
  public static BufferedImage randomize(final BufferedImage contentImage) {return randomize(contentImage, x -> FastRandom.INSTANCE.random());}
  
  /**
   * Randomize buffered image.
   *
   * @param contentImage the content image
   * @param f            the f
   * @return the buffered image
   */
  @Nonnull
  public static BufferedImage randomize(final BufferedImage contentImage, final DoubleUnaryOperator f) {
    return Tensor.fromRGB(contentImage).map(f).toRgbImage();
  }
  
  /**
   * Paint noise.
   *
   * @param canvas the canvas
   */
  public static void paint_noise(final Tensor canvas) {
    canvas.setByCoord(c -> FastRandom.INSTANCE.random());
  }
  
  /**
   * Wrap tiles avg layer.
   *
   * @param subnet the subnet
   * @return the layer
   */
  protected static Layer wrapTilesAvg(final Layer subnet) {
    return wrapTilesAvg(subnet, 0, 0, 0, 0, 400, 400);
  }
  
  /**
   * Wrap tiles avg layer.
   *
   * @param subnet     the subnet
   * @param borderX1   the border x 1
   * @param borderY1   the border y 1
   * @param borderX2   the border x 2
   * @param borderY2   the border y 2
   * @param tileWidth  the tile width
   * @param tileHeight the tile height
   * @return the layer
   */
  protected static Layer wrapTilesAvg(final Layer subnet, final int borderX1, final int borderY1, final int borderX2, final int borderY2, final int tileWidth, final int tileHeight) {
    PipelineNetwork network1 = new PipelineNetwork(1);
    if (borderX1 != 0 || borderY1 != 0)
      network1.wrap(new com.simiacryptus.mindseye.layers.cudnn.ImgZeroPaddingLayer(borderX1, borderY1));
    network1.add(subnet);
    if (borderX2 != 0 || borderY2 != 0)
      network1.wrap(new com.simiacryptus.mindseye.layers.cudnn.ImgZeroPaddingLayer(-borderX2, -borderY2));
    PipelineNetwork network = new PipelineNetwork(1);
    network.wrap(new com.simiacryptus.mindseye.layers.cudnn.ImgTileSubnetLayer(network1, tileWidth, tileHeight, tileWidth - 2 * borderX1, tileHeight - 2 * borderY1));
    network.wrap(new BandAvgReducerLayer());
    return network;
  }
  
  /**
   * Log exception with default t.
   *
   * @param <T>          the type parameter
   * @param log          the log
   * @param fn           the fn
   * @param defaultValue the default value
   * @return the t
   */
  public static <T> T logExceptionWithDefault(@Nonnull final NotebookOutput log, Supplier<T> fn, T defaultValue) {
    try {
      return fn.get();
    } catch (Throwable throwable) {
      try {
        log.code(() -> {
          return throwable;
        });
      } catch (Throwable e2) {
      }
      return defaultValue;
    }
  }
}
