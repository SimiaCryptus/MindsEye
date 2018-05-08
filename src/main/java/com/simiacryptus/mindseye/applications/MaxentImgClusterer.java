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
import com.simiacryptus.mindseye.network.InnerNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.opt.IterativeTrainer;
import com.simiacryptus.mindseye.opt.line.BisectionSearch;
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

import javax.annotation.Nonnull;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class MaxentImgClusterer {
  private static final Logger logger = LoggerFactory.getLogger(MaxentImgClusterer.class);
  private final int clusters;
  private final double seedPcaPower;
  private final int maxRetries;
  private final int orientation;
  private final double globalDistributionEmphasis;
  private final double selectionEntropyAdj;
  private final int maxIterations;
  private final int timeoutMinutes;
  
  MaxentImgClusterer(final int clusters, final int maxRetries, final int orientation, final double globalDistributionEmphasis, final double selectionEntropyAdj, final int maxIterations, final int timeoutMinutes, final double seedPcaPower) {
    this.clusters = clusters;
    this.maxRetries = maxRetries;
    this.orientation = orientation;
    this.globalDistributionEmphasis = globalDistributionEmphasis;
    this.selectionEntropyAdj = selectionEntropyAdj;
    this.maxIterations = maxIterations;
    this.timeoutMinutes = timeoutMinutes;
    this.seedPcaPower = seedPcaPower;
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
  
  public static List<Tensor> pca(final double[] bandCovariance, final double eigenPower) {
    @Nonnull final EigenDecomposition decomposition = new EigenDecomposition(toMatrix(bandCovariance));
    return IntStream.range(0, (int) Math.sqrt(bandCovariance.length)).mapToObj(vectorIndex -> {
      double[] data = decomposition.getEigenvector(vectorIndex).toArray();
      return new Tensor(data, 1, 1, data.length).scale(Math.pow(decomposition.getRealEigenvalue(vectorIndex), eigenPower));
    }).collect(Collectors.toList());
  }
  
  public static double[] getPixel(final Tensor tensor, final int x, final int y, final int bands) {
    return IntStream.range(0, bands).mapToDouble(band -> tensor.get(x, y, band)).toArray();
  }
  
  public static int countPixels(final Tensor featureImage) {
    int[] dimensions = featureImage.getDimensions();
    int width = dimensions[0];
    int height = dimensions[1];
    return width * height;
  }
  
  @Nonnull
  public static Array2DRowRealMatrix toMatrix(final double[] covariance) {
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
  
  public Layer analyze(final Tensor metricsImage, final NotebookOutput log) {
    
    double[] mean = new BandReducerLayer().setMode(PoolingLayer.PoolingMode.Avg).eval(metricsImage).getDataAndFree().getAndFree(0).getData();
    double[] bandCovariance = bandCovariance(metricsImage.getPixelStream(), countPixels(metricsImage), mean);
    List<Tensor> seedVectors = pca(bandCovariance, getSeedPcaPower()).stream().collect(Collectors.toList());
    
    Layer metricsToClusters = null;
    int[] dimensions = metricsImage.getDimensions();
    int pixels = dimensions[0] * dimensions[1];
    int bands = dimensions[2];
    String convolutionLayerName = "mix";
    for (int retries = 0; retries < getMaxRetries(); retries++) {
      ConvolutionLayer convolutionLayer = new ConvolutionLayer(1, 1, bands, getClusters());
      convolutionLayer.getKernel().setByCoord(c -> {
        int band = c.getCoords()[2];
//        int index2 = band % getClusters();
//        int index1 = band / getClusters();
        int index1 = band % bands;
        int index2 = band / bands;
        return seedVectors.get(index2 % seedVectors.size()).get(index1) * ((index2 < seedVectors.size()) ? 1 : -1);
      });
      metricsToClusters = PipelineNetwork.build(1,
        new ImgBandBiasLayer(bands).set(new Tensor(mean).scale(-1)),
        convolutionLayer.explode().setName(convolutionLayerName),
        new SoftmaxActivationLayer().setMode(SoftmaxActivationLayer.SoftmaxMode.CHANNEL).setAlgorithm(SoftmaxActivationLayer.SoftmaxAlgorithm.ACCURATE)
      );
      
      PipelineNetwork netEntropy = new PipelineNetwork(1);
      InnerNode membership = netEntropy.wrap(metricsToClusters, netEntropy.getInput(0));
      netEntropy.wrap(new BinarySumLayer(getOrientation(), getOrientation() * -Math.pow(2, getGlobalDistributionEmphasis())),
        netEntropy.wrap(PipelineNetwork.build(1,
          new SoftmaxActivationLayer().setMode(SoftmaxActivationLayer.SoftmaxMode.CHANNEL).setAlgorithm(SoftmaxActivationLayer.SoftmaxAlgorithm.ACCURATE),
          new AutoEntropyLayer()
        ), membership),
        netEntropy.wrap(PipelineNetwork.build(1,
          new ScaleLayer(new Tensor(Math.pow(2, getSelectionEntropyAdj()))),
          new SoftmaxActivationLayer().setMode(SoftmaxActivationLayer.SoftmaxMode.CHANNEL).setAlgorithm(SoftmaxActivationLayer.SoftmaxAlgorithm.ACCURATE),
          new BandAvgReducerLayer().setAlpha(pixels),
          new AutoEntropyLayer()
        ), membership));
      Trainable trainable = new ArrayTrainable(netEntropy, 1).setVerbose(true).setMask(false).setData(Arrays.asList(new Tensor[][]{{metricsImage}}));
      
      @Nonnull ArrayList<StepRecord> history = new ArrayList<>();
      try {
        log.code(() -> {
          new IterativeTrainer(trainable)
            .setMonitor(TestUtil.getMonitor(history))
            .setIterationsPerSample(100)
            .setOrientation(new TrustRegionStrategy() {
              @Override
              public TrustRegion getRegionPolicy(final Layer layer) {
                if (layer instanceof SimpleConvolutionLayer) return new RangeConstraint(-1, 1);
                return null;
              }
            })
            .setMaxIterations(getMaxIterations())
            .setLineSearchFactory(name -> new QuadraticSearch().setRelativeTolerance(1e-1))
            .setLineSearchFactory(name -> new BisectionSearch().setSpanTol(1e-1).setCurrentRate(1e-1))
            .setTimeout(getTimeoutMinutes(), TimeUnit.MINUTES)
            .setTerminateThreshold(Double.NEGATIVE_INFINITY)
            .runAndFree();
        });
        log.code(() -> {
          return TestUtil.plot(history);
        });
        break;
      } catch (Throwable e) {
        logger.info("Error training clustering network", e);
      }
    }
    return PipelineNetwork.build(1,
      metricsToClusters,
      new SoftmaxActivationLayer().setMode(SoftmaxActivationLayer.SoftmaxMode.CHANNEL).setAlgorithm(SoftmaxActivationLayer.SoftmaxAlgorithm.ACCURATE)
    );
  }
  
  public int getClusters() {
    return clusters;
  }
  
  public int getMaxRetries() {
    return maxRetries;
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
    return "MaxentImgClusterer{" +
      "clusters=" + clusters +
      ", seedPcaPower=" + seedPcaPower +
      ", maxRetries=" + maxRetries +
      ", orientation=" + orientation +
      ", globalDistributionEmphasis=" + globalDistributionEmphasis +
      ", selectionEntropyAdj=" + selectionEntropyAdj +
      ", maxIterations=" + maxIterations +
      ", timeoutMinutes=" + timeoutMinutes +
      '}';
  }
  
  public double getSeedPcaPower() {
    return seedPcaPower;
  }
}
