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

import com.simiacryptus.mindseye.eval.ArrayTrainable;
import com.simiacryptus.mindseye.eval.Trainable;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.RecycleBin;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.cudnn.BandAvgReducerLayer;
import com.simiacryptus.mindseye.layers.cudnn.BandReducerLayer;
import com.simiacryptus.mindseye.layers.cudnn.BinarySumLayer;
import com.simiacryptus.mindseye.layers.cudnn.ImgBandBiasLayer;
import com.simiacryptus.mindseye.layers.cudnn.PoolingLayer;
import com.simiacryptus.mindseye.layers.cudnn.ScaleLayer;
import com.simiacryptus.mindseye.layers.cudnn.SoftmaxActivationLayer;
import com.simiacryptus.mindseye.layers.cudnn.conv.ConvolutionLayer;
import com.simiacryptus.mindseye.layers.cudnn.conv.SimpleConvolutionLayer;
import com.simiacryptus.mindseye.layers.java.AutoEntropyLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.opt.IterativeTrainer;
import com.simiacryptus.mindseye.opt.line.QuadraticSearch;
import com.simiacryptus.mindseye.opt.orient.TrustRegionStrategy;
import com.simiacryptus.mindseye.opt.region.RangeConstraint;
import com.simiacryptus.mindseye.opt.region.TrustRegion;
import com.simiacryptus.mindseye.test.StepRecord;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.util.io.NotebookOutput;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.EigenDecomposition;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import smile.plot.PlotCanvas;

import javax.annotation.Nonnull;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class PixelClusterer {
  private static final Logger logger = LoggerFactory.getLogger(PixelClusterer.class);
  private int clusters;
  private double seedPcaPower;
  private int orientation;
  private double globalDistributionEmphasis;
  private double selectionEntropyAdj;
  private int maxIterations;
  private int timeoutMinutes;
  private double seedMagnitude;
  
  public PixelClusterer(final int clusters, final int orientation, final double globalDistributionEmphasis, final double selectionEntropyAdj, final int maxIterations, final int timeoutMinutes, final double seedPcaPower, final double seedMagnitude) {
    this.setClusters(clusters);
    this.setOrientation(orientation);
    this.setGlobalDistributionEmphasis(globalDistributionEmphasis);
    this.setSelectionEntropyAdj(selectionEntropyAdj);
    this.setMaxIterations(maxIterations);
    this.setTimeoutMinutes(timeoutMinutes);
    this.setSeedPcaPower(seedPcaPower);
    this.setSeedMagnitude(seedMagnitude);
  }
  
  public PixelClusterer(final int clusters) {
    this(
      clusters,
      -1,
      0,
      0,
      10,
      10,
      0.5,
      1e-2
    );
  }
  
  public static double[] bandCovariance(final Stream<double[]> pixelStream, final int pixels, final double[] mean) {
    return Arrays.stream(pixelStream.map(pixel -> {
      double[] crossproduct = RecycleBin.DOUBLES.obtain(pixel.length * pixel.length);
      int k = 0;
      for (int j = 0; j < pixel.length; j++) {
        for (int i = 0; i < pixel.length; i++) {
          crossproduct[k++] = (pixel[i] - mean[i]) * (pixel[j] - mean[j]);
        }
      }
      RecycleBin.DOUBLES.recycle(pixel, pixel.length);
      return crossproduct;
    }).reduce((a, b) -> {
      for (int i = 0; i < a.length; i++) {
        a[i] += b[i];
      }
      RecycleBin.DOUBLES.recycle(b, b.length);
      return a;
    }).get()).map(x -> x / pixels).toArray();
  }
  
  private static List<Tensor> pca(final double[] bandCovariance, final double eigenPower) {
    @Nonnull final EigenDecomposition decomposition = new EigenDecomposition(toMatrix(bandCovariance));
    return IntStream.range(0, (int) Math.sqrt(bandCovariance.length)).mapToObj(vectorIndex -> {
      double[] data = decomposition.getEigenvector(vectorIndex).toArray();
      return new Tensor(data, 1, 1, data.length).scale(Math.pow(decomposition.getRealEigenvalue(vectorIndex), eigenPower));
    }).collect(Collectors.toList());
  }
  
  public static double[] getPixel(final Tensor tensor, final int x, final int y, final int bands) {
    return IntStream.range(0, bands).mapToDouble(band -> tensor.get(x, y, band)).toArray();
  }
  
  private static int countPixels(final Tensor featureImage) {
    int[] dimensions = featureImage.getDimensions();
    int width = dimensions[0];
    int height = dimensions[1];
    return width * height;
  }
  
  @Nonnull
  private static Array2DRowRealMatrix toMatrix(final double[] covariance) {
    final int bands = (int) Math.sqrt(covariance.length);
    Array2DRowRealMatrix matrix = new Array2DRowRealMatrix(bands, bands);
    int k = 0;
    for (int x = 0; x < bands; x++) {
      for (int y = 0; y < bands; y++) {
        matrix.setEntry(x, y, covariance[k++]);
      }
    }
    return matrix;
  }
  
  public Layer analyze(final Tensor metrics) {
    Layer model = modelingNetwork(metrics);
    int[] dimensions = metrics.getDimensions();
    train(getTrainable(metrics, model.andThen(entropyNetwork(dimensions[0] * dimensions[1]))));
    return model.andThen(new SoftmaxActivationLayer().setMode(SoftmaxActivationLayer.SoftmaxMode.CHANNEL));
  }
  
  public Layer analyze(final NotebookOutput log, final Tensor metrics) {
    Layer model = modelingNetwork(metrics);
    log.code(() -> {
      int[] dimensions = metrics.getDimensions();
      return train(getTrainable(metrics, model.andThen(entropyNetwork(dimensions[0] * dimensions[1]))));
    });
    return model.andThen(new SoftmaxActivationLayer().setMode(SoftmaxActivationLayer.SoftmaxMode.CHANNEL));
  }
  
  public Layer modelingNetwork(final Tensor metrics) {
    int[] dimensions = metrics.getDimensions();
    int bands = dimensions[2];
    double[] mean = new BandReducerLayer().setMode(PoolingLayer.PoolingMode.Avg).eval(metrics).getDataAndFree().getAndFree(0).getData();
    double[] bandCovariance = bandCovariance(metrics.getPixelStream(), countPixels(metrics), mean);
    List<Tensor> seedVectors = pca(bandCovariance, getSeedPcaPower()).stream().collect(Collectors.toList());
    String convolutionLayerName = "mix";
    ConvolutionLayer convolutionLayer = new ConvolutionLayer(1, 1, bands, getClusters());
    convolutionLayer.getKernel().setByCoord(c -> {
      int band = c.getCoords()[2];
//        int index1 = band / getClusters();
//        int index2 = band % getClusters();
      int index1 = band % bands;
      int index2 = band / bands;
      return getSeedMagnitude() * seedVectors.get(index2 % seedVectors.size()).get(index1) * ((index2 < seedVectors.size()) ? 1 : -1);
    });
    return PipelineNetwork.build(1,
      new ImgBandBiasLayer(bands).set(new Tensor(mean).scale(-1)),
      convolutionLayer.explode().setName(convolutionLayerName)
    );
  }
  
  @Nonnull
  public Trainable getTrainable(final Tensor metrics, final Layer netEntropy) {
    return new ArrayTrainable(netEntropy, 1).setVerbose(true).setMask(false).setData(Arrays.asList(new Tensor[][]{{metrics}}));
  }
  
  @Nonnull
  public Layer entropyNetwork(final int pixels) {
    PipelineNetwork netEntropy = new PipelineNetwork(1);
    netEntropy.wrap(new BinarySumLayer(getOrientation(), getOrientation() * -Math.pow(2, getGlobalDistributionEmphasis())),
      netEntropy.wrap(PipelineNetwork.build(1,
        new SoftmaxActivationLayer().setMode(SoftmaxActivationLayer.SoftmaxMode.CHANNEL).setAlgorithm(SoftmaxActivationLayer.SoftmaxAlgorithm.ACCURATE),
        new AutoEntropyLayer()
      ), netEntropy.getInput(0)),
      netEntropy.wrap(PipelineNetwork.build(1,
        new ScaleLayer(new Tensor(Math.pow(2, getSelectionEntropyAdj()))),
        new SoftmaxActivationLayer().setMode(SoftmaxActivationLayer.SoftmaxMode.CHANNEL).setAlgorithm(SoftmaxActivationLayer.SoftmaxAlgorithm.ACCURATE),
        new BandAvgReducerLayer().setAlpha(pixels),
        new AutoEntropyLayer()
      ), netEntropy.getInput(0)));
    return netEntropy;
  }
  
  public PlotCanvas train(final Trainable trainable) {
    @Nonnull ArrayList<StepRecord> history = new ArrayList<>();
    try {
      new IterativeTrainer(trainable)
        .setMonitor(TestUtil.getMonitor(history))
        .setOrientation(new TrustRegionStrategy() {
          @Override
          public TrustRegion getRegionPolicy(final Layer layer) {
            if (layer instanceof SimpleConvolutionLayer) return new RangeConstraint(-1, 1);
            return null;
          }
        })
        .setMaxIterations(getMaxIterations())
        .setLineSearchFactory(name -> new QuadraticSearch().setRelativeTolerance(1e-1))
        //.setLineSearchFactory(name -> new BisectionSearch().setSpanTol(1e-1).setCurrentRate(1e-1))
        .setTimeout(getTimeoutMinutes(), TimeUnit.MINUTES)
        .setTerminateThreshold(Double.NEGATIVE_INFINITY)
        .runAndFree();
    } catch (Throwable e) {
      logger.warn("Error training", e);
    } finally {
      return TestUtil.plot(history);
    }
  }
  
  public int getClusters() {
    return clusters;
  }
  
  public int getOrientation() {
    return orientation;
  }
  
  public double getGlobalDistributionEmphasis() {
    return globalDistributionEmphasis;
  }
  
  public double getSelectionEntropyAdj() {
    return selectionEntropyAdj;
  }
  
  public int getMaxIterations() {
    return maxIterations;
  }
  
  public int getTimeoutMinutes() {
    return timeoutMinutes;
  }
  
  @Override
  public String toString() {
    return getClass().getSimpleName() + "{" +
      "clusters=" + getClusters() +
      ", seedPcaPower=" + getSeedPcaPower() +
      ", orientation=" + getOrientation() +
      ", globalDistributionEmphasis=" + getGlobalDistributionEmphasis() +
      ", selectionEntropyAdj=" + getSelectionEntropyAdj() +
      ", maxIterations=" + getMaxIterations() +
      ", timeoutMinutes=" + getTimeoutMinutes() +
      '}';
  }
  
  public double getSeedPcaPower() {
    return seedPcaPower;
  }
  
  public double getSeedMagnitude() {
    return seedMagnitude;
  }
  
  public PixelClusterer setSeedMagnitude(double seedMagnitude) {
    this.seedMagnitude = seedMagnitude;
    return this;
  }
  
  public PixelClusterer setClusters(int clusters) {
    this.clusters = clusters;
    return this;
  }
  
  public PixelClusterer setSeedPcaPower(double seedPcaPower) {
    this.seedPcaPower = seedPcaPower;
    return this;
  }
  
  public PixelClusterer setOrientation(int orientation) {
    this.orientation = orientation;
    return this;
  }
  
  public PixelClusterer setGlobalDistributionEmphasis(double globalDistributionEmphasis) {
    this.globalDistributionEmphasis = globalDistributionEmphasis;
    return this;
  }
  
  public PixelClusterer setSelectionEntropyAdj(double selectionEntropyAdj) {
    this.selectionEntropyAdj = selectionEntropyAdj;
    return this;
  }
  
  public PixelClusterer setMaxIterations(int maxIterations) {
    this.maxIterations = maxIterations;
    return this;
  }
  
  public PixelClusterer setTimeoutMinutes(int timeoutMinutes) {
    this.timeoutMinutes = timeoutMinutes;
    return this;
  }
}
