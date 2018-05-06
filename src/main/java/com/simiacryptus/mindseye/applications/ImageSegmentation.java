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

import com.simiacryptus.mindseye.lang.DeltaSet;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.MutableResult;
import com.simiacryptus.mindseye.lang.RecycleBin;
import com.simiacryptus.mindseye.lang.Result;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.layers.cudnn.BandReducerLayer;
import com.simiacryptus.mindseye.layers.cudnn.ImgBandBiasLayer;
import com.simiacryptus.mindseye.layers.cudnn.PoolingLayer;
import com.simiacryptus.mindseye.layers.cudnn.ProductLayer;
import com.simiacryptus.mindseye.layers.cudnn.SquareActivationLayer;
import com.simiacryptus.mindseye.layers.cudnn.conv.SimpleConvolutionLayer;
import com.simiacryptus.mindseye.layers.java.ImgPixelSumLayer;
import com.simiacryptus.mindseye.layers.java.NthPowerActivationLayer;
import com.simiacryptus.mindseye.layers.java.SumReducerLayer;
import com.simiacryptus.mindseye.models.CVPipe;
import com.simiacryptus.mindseye.models.CVPipe_VGG19;
import com.simiacryptus.mindseye.models.LayerEnum;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.util.io.NotebookOutput;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.clustering.Clusterable;
import org.apache.commons.math3.ml.clustering.Clusterer;
import org.apache.commons.math3.ml.clustering.DoublePoint;
import org.apache.commons.math3.ml.clustering.KMeansPlusPlusClusterer;
import org.apache.commons.math3.ml.distance.DistanceMeasure;
import org.apache.commons.math3.ml.distance.EuclideanDistance;
import org.apache.commons.math3.random.JDKRandomGenerator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import java.util.Arrays;
import java.util.Collection;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * The type Object location.
 */
public abstract class ImageSegmentation<T extends LayerEnum<T>, U extends CVPipe<T>> {
  
  private static final Logger logger = LoggerFactory.getLogger(ImageSegmentation.class);
  
  public static List<Tensor> refineMasks_pca_0(final List<Tensor> normalizedFeatureMasks) {
    double[] covariance = normalizedFeatureMasks.stream().parallel()
      .flatMapToDouble(x -> normalizedFeatureMasks.stream().mapToDouble(y -> x.dot(y))).toArray();
    return pca(covariance).stream()
      .map(eigenvector -> linearMix(normalizedFeatureMasks, eigenvector.getData()).map(v -> Math.abs(v)))
      .collect(Collectors.toList());
  }
  
  public static List<Tensor> preblur(final List<Tensor> featureMasks, final int blurIterations) {
    return featureMasks.stream().parallel()
      .map(x -> x.scale(1.0 / x.l2()))
      .map(scale -> PCAObjectLocation.blur(scale, blurIterations))
      .map(PCAObjectLocation::sum3channels)
      //.collect(Collectors.toList()).stream()
      .map(x -> {
        double meanV = x.mean();
        return x.map(v -> v - meanV);
      })
      .map(x -> x.scale(1.0 / x.l2()))
      .collect(Collectors.toList());
  }
  
  public static List<Tensor> refineMasks_pca(final List<Tensor> inputMasks) {
    assert inputMasks.stream().allMatch(x -> x.getDimensions().length == 3);
    assert inputMasks.stream().allMatch(x -> x.getDimensions()[2] == 1);
    Tensor maskImg = unitImageNetwork().eval(concat(inputMasks)).getDataAndFree().get(0);
    return pca(maskImg).stream().map(vector -> linearMix(inputMasks, vector.getData())).collect(Collectors.toList());
  }
  
  public static List<Tensor> primaryFeatureVectors_simple(final int vectorSamples, final Tensor featureImage) {
    int[] dimensions = featureImage.getDimensions();
    int bands = dimensions[2];
    double[] mean = new BandReducerLayer().setMode(PoolingLayer.PoolingMode.Avg).eval(featureImage).getDataAndFree().getAndFree(0).getData();
    double[] avgSq = getAvgSq(featureImage, mean);
    int[] sortedIndices = IntStream.range(0, mean.length).mapToObj(x -> x).sorted(Comparator.comparing(i ->
      -Math.abs(avgSq[i])
    )).mapToInt(x -> x).toArray();
    return IntStream.range(0, Math.min(vectorSamples, sortedIndices.length)).mapToObj(i -> {
      Tensor tensor = new Tensor(1, 1, bands);
      tensor.set(sortedIndices[i], 1);
      return tensor;
    }).collect(Collectors.toList());
  }
  
  public static List<Tensor> refineMasks_kmeans(final List<Tensor> masks) {
    assert masks.stream().allMatch(x -> x.getDimensions().length == 3);
    assert masks.stream().allMatch(x -> x.getDimensions()[2] == 1);
    int[] dimensions = masks.get(0).getDimensions();
    Tensor maskConcat = concat(masks);
    Clusterer<DoublePoint> clusterer = new KMeansPlusPlusClusterer<>(masks.size(), -1, new EuclideanDistance(), new JDKRandomGenerator(), KMeansPlusPlusClusterer.EmptyClusterStrategy.LARGEST_POINTS_NUMBER);
//    Clusterer<DoublePoint> clusterer = new FuzzyKMeansClusterer<DoublePoint>(vectorSamples + 2, 2.0);
    List<CentroidCluster<DoublePoint>> clusters = clusterer
      .cluster(getPixelStream(maskConcat).map(DoublePoint::new).collect(Collectors.toList()))
      .stream()
      .sorted(Comparator.comparing(x1 -> -x1.getPoints().size()))
      //.collect(Collectors.toList()).stream()
      .map(x -> (CentroidCluster<DoublePoint>) x)
      .collect(Collectors.toList());
    
    Tensor image = new Tensor(dimensions[0], dimensions[1], clusters.size());
    for (int x = 0; x < dimensions[0]; x++) {
      for (int y = 0; y < dimensions[1]; y++) {
        double[] pixel = getPixel(maskConcat, x, y);
        double[] dist = dist(clusters, clusterer.getDistanceMeasure(), pixel);
        for (int band = 0; band < dist.length; band++) {
          image.set(x, y, band, dist[band]);
        }
      }
    }
    return splitBands(unitImageNetwork().eval(image).getDataAndFree().get(0));
//    return splitBands(image);
  }
  
  public static double[] getPixel(final Tensor tensor, final int x, final int y) {
    return getPixel(tensor, x, y, tensor.getDimensions()[2]);
  }
  
  public static double[] getPixel(final Tensor tensor, final int x, final int y, final int bands) {
    return IntStream.range(0, bands).mapToDouble(band -> tensor.get(x, y, band)).toArray();
  }
  
  public static Tensor concat(final List<Tensor> featureImage) {
    assert featureImage.stream().map(Tensor::getDimensions).allMatch(dims -> dims.length == 2 || (dims.length == 3 && dims[2] == 1));
    int[] dimensions = featureImage.get(0).getDimensions();
    Tensor image = new Tensor(dimensions[0], dimensions[1], featureImage.size());
    image.setByCoord(c -> {
      int[] coords = c.getCoords();
      return featureImage.get(coords[2]).get(coords[0], coords[1]);
    });
    return image;
  }
  
  public static Tensor linearMix(final List<Tensor> inputMasks, final double... data) {
    return IntStream.range(0, inputMasks.size()).mapToObj(i -> inputMasks.get(i).scale(data[i])).reduce((a, b) -> a.add(b)).get().scale(1.0 / inputMasks.size());
  }
  
  public static List<Tensor> refineMasks_echo(final List<Tensor> inputMasks) {
    assert inputMasks.stream().allMatch(x -> x.getDimensions().length == 3);
    assert inputMasks.stream().allMatch(x -> x.getDimensions()[2] == 1);
    Tensor maskImg = concat(inputMasks);
    //maskImg = unitImageNetwork().eval(maskImg).getDataAndFree().get(0);
    //pca(vectorSamples, maskImg);
    return splitBands(maskImg);
  }
  
  public static List<Tensor> splitBands(final Tensor image) {
    return IntStream.range(0, image.getDimensions()[2]).mapToObj(i -> {
      return image.selectBand(i);
    }).collect(Collectors.toList());
  }
  
  public static double[] dist(final List<CentroidCluster<DoublePoint>> clusters, final DistanceMeasure measure, final double[] pixel) {
    return IntStream.range(0, clusters.size())
      .mapToObj(clusters::get)
      .map(CentroidCluster::getCenter)
      .map(Clusterable::getPoint)
      .mapToDouble(centroid -> measure.compute(centroid, pixel))
      .toArray();
  }
  
  public static double[] unitV(final double[] vector) {
    double magnitude = Math.sqrt(Arrays.stream(vector).map(v -> v * v).sum());
    return Arrays.stream(vector).map(v -> v / magnitude).toArray();
  }
  
  public static List<Tensor> primaryFeatureVectors_kmeans(final int vectorSamples, final Tensor featureImage) {
    return kmeans(vectorSamples, getPixelStream(featureImage).collect(Collectors.toList()));
  }
  
  public static List<Tensor> kmeans(final int clusters, final Collection<double[]> pixels) {
    List<DoublePoint> points = pixels.stream().map(DoublePoint::new).collect(Collectors.toList());
    Clusterer<DoublePoint> clusterer = new KMeansPlusPlusClusterer<>(clusters, -1, new EuclideanDistance(), new JDKRandomGenerator(), KMeansPlusPlusClusterer.EmptyClusterStrategy.LARGEST_POINTS_NUMBER);
//    Clusterer<DoublePoint> clusterer = new FuzzyKMeansClusterer<DoublePoint>(vectorSamples + 2, 2.0);
    return clusterer
      .cluster(points)
      .stream()
      .sorted(Comparator.comparing(x1 -> -x1.getPoints().size()))
      .limit(clusters)
      //.collect(Collectors.toList()).stream()
      .map(x -> (CentroidCluster<DoublePoint>) x)
      .map(CentroidCluster::getCenter)
      .map(Clusterable::getPoint)
      .map(Tensor::new)
      .collect(Collectors.toList());
  }
  
  public static double[] cov(final List<double[]> pixels, final double[] mean) {
    final int bands = mean.length;
    return Arrays.stream(pixels.stream().map(pixel -> {
      double[] crossproduct = RecycleBin.DOUBLES.obtain(bands * bands);
      int k = 0;
      for (int j = 0; j < bands; j++) {
        for (int i = 0; i < bands; i++) {
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
    }).get()).map(x -> x / (pixels.size())).toArray();
  }
  
  public static List<Tensor> pca(final Tensor featureImage) {
    double[] mean = new BandReducerLayer().setMode(PoolingLayer.PoolingMode.Avg).eval(featureImage).getDataAndFree().getAndFree(0).getData();
    double[] bandCovariance = bandCovariance(getPixelStream(featureImage), countPixels(featureImage), mean);
    return pca(bandCovariance);
  }
  
  @Nonnull
  public static Tensor getDeltaMaskFromFeatureVector(final int[] imgDimensions, final Result featureResult, final Tensor eigenvector) {
    SimpleConvolutionLayer metricFilter = new SimpleConvolutionLayer(1, 1, eigenvector.length());
    metricFilter.set(eigenvector);
    metricFilter.freeze();
    DeltaSet<Layer> deltaBuffer = new DeltaSet<>();
    PipelineNetwork.build(1, unitImageNetwork(), metricFilter, new SquareActivationLayer(), new SumReducerLayer()).eval(featureResult).accumulate(deltaBuffer);
    if (deltaBuffer.getMap().size() != 1) throw new AssertionError(deltaBuffer.getMap().size());
    double[] delta = deltaBuffer.getMap().values().iterator().next().getDelta();
    return new Tensor(delta, imgDimensions[0], imgDimensions[1], 3)
      .map(v -> Math.abs(v))
      ;
  }
  
  @Nonnull
  public static PipelineNetwork unitImageNetwork() {
    PipelineNetwork pipelineNetwork = PipelineNetwork.build(1, new SquareActivationLayer(), new ImgPixelSumLayer(), new NthPowerActivationLayer().setPower(-0.5));
    pipelineNetwork.add(new ProductLayer(), pipelineNetwork.getInput(0), pipelineNetwork.getHead());
    return pipelineNetwork;
  }
  
  @Nonnull
  public static PipelineNetwork normalizeBandsNetwork() {
    PipelineNetwork pipelineNetwork = PipelineNetwork.build(1, new SquareActivationLayer(), new ImgPixelSumLayer(), new NthPowerActivationLayer().setPower(-0.5));
    pipelineNetwork.add(new ProductLayer(), pipelineNetwork.getInput(0), pipelineNetwork.getHead());
    return pipelineNetwork;
  }
  
  public static List<Tensor> pca(final double[] bandCovariance) {
    @Nonnull final EigenDecomposition decomposition = new EigenDecomposition(toMatrix(bandCovariance));
    return IntStream.range(0, (int) Math.sqrt(bandCovariance.length)).mapToObj(vectorIndex -> {
      double[] data = decomposition.getEigenvector(vectorIndex).toArray();
      return new Tensor(data, 1, 1, data.length).scale(decomposition.getRealEigenvalue(vectorIndex));
    }).collect(Collectors.toList());
  }
  
  public static void display(@Nonnull final NotebookOutput log, final Tensor img, Tensor maskData) {
    if (maskData.getDimensions()[2] == 3) {
      display(log, img, PCAObjectLocation.sum3channels(maskData));
    }
    else {
      maskData = maskData.scale(255.0 / maskData.rms());
      //    log.p("EigenVector (raw): " + Arrays.toString(eigenvector));
      //    log.p("EigenVector (sorted): " + Arrays.toString(Arrays.stream(eigenvector).sorted().toArray()));
      log.p(log.image(img.toRgbImageAlphaMask(0, 1, 2, maskData), ""));
    }
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
  
  @Nonnull
  public static Stream<double[]> getPixelStream(final Tensor featureImage) {
    int[] dimensions = featureImage.getDimensions();
    int width = dimensions[0];
    int height = dimensions[1];
    int bands = dimensions[2];
    return IntStream.range(0, width).mapToObj(x -> x).parallel().flatMap(x -> {
      return IntStream.range(0, height).mapToObj(y -> y).map(y -> {
        return getPixel(featureImage, x, y, bands);
      });
    });
  }
  
  public static double[] getAvg(final Tensor... featureImage) {
    return getAvg(Arrays.asList(featureImage));
  }
  
  public static double[] getAvg(final Collection<Tensor> featureImage) {
    return featureImage.stream().map(tensor -> {
      return new BandReducerLayer().setMode(PoolingLayer.PoolingMode.Avg).eval(
        new SquareActivationLayer().eval(tensor)
      ).getDataAndFree().getAndFree(0);
    }).reduce((a, b) -> a.add(b)).get().scale(1.0 / featureImage.size()).getData();
  }
  
  public static double[] getAvgSq(final Tensor featureImage, final double[] mean) {
    return new BandReducerLayer().setMode(PoolingLayer.PoolingMode.Avg).eval(
      new SquareActivationLayer().eval(new ImgBandBiasLayer(mean.length).set(new Tensor(mean).scale(-1)).eval(featureImage))
    ).getDataAndFree().getAndFree(0).getData();
  }
  
  public static double[] bandCovariance(final Tensor featureImage, final double[] mean, final double[] avgSqBands) {
    return bandCovariance(getPixelStream(featureImage), countPixels(featureImage), mean, avgSqBands);
  }
  
  public static double[] bandCovariance(final Stream<double[]> pixelStream, final int pixels, final double[] mean, final double[] avgSqBands) {
    return Arrays.stream(pixelStream.map(pixel -> {
      double[] crossproduct = RecycleBin.DOUBLES.obtain(pixel.length * pixel.length);
      int k = 0;
      for (int j = 0; j < pixel.length; j++) {
        for (int i = 0; i < pixel.length; i++) {
          crossproduct[k++] = ((pixel[i] - mean[i]) / Math.sqrt(avgSqBands[i])) * ((pixel[j] - mean[j]) / Math.sqrt(avgSqBands[j]));
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
  
  /**
   * Run.
   *
   * @param log the log
   */
  public void run(@Nonnull final NotebookOutput log, Tensor img) {
//    run(log, img, 5,
//      CVPipe_VGG19.Layer.Layer_0,
//      CVPipe_VGG19.Layer.Layer_1a,
//      CVPipe_VGG19.Layer.Layer_1b,
//      CVPipe_VGG19.Layer.Layer_1c,
//      CVPipe_VGG19.Layer.Layer_1d,
//      CVPipe_VGG19.Layer.Layer_1e
//    );
    run(log, img, 5,
      CVPipe_VGG19.Layer.Layer_0,
      CVPipe_VGG19.Layer.Layer_1a,
      //CVPipe_VGG19.Layer.Layer_1b,
      //CVPipe_VGG19.Layer.Layer_1c,
      CVPipe_VGG19.Layer.Layer_1d
      //CVPipe_VGG19.Layer.Layer_1e
    );
  }
  
  public void run(@Nonnull final NotebookOutput log, final Tensor img, final int vectorSamples, final CVPipe_VGG19.Layer... layers) {
    log.p(log.image(img.toImage(), ""));
    
    List<Tensor> featureMasks = Arrays.stream(getLayerTypes()).filter(x -> Arrays.asList(layers).contains(x)).flatMap(layer -> {
      log.h3(layer.name());
      Layer network = getInstance().getPrototypes().get(layer);
      ArtistryUtil.setPrecision((DAGNetwork) network, Precision.Float);
      network.setFrozen(true);
      Result imageFeatures = network.eval(new MutableResult(img));
      Tensor featureImage = imageFeatures.getData().get(0);
      int[] dimensions = featureImage.getDimensions();
      int bands = dimensions[2];
      log.p("Feature Image Dimension: " + Arrays.toString(dimensions));
      return pca(featureImage).stream().limit(vectorSamples).map(eigenvector -> {
        Tensor maskData = getDeltaMaskFromFeatureVector(img.getDimensions(), imageFeatures, eigenvector);
        maskData = maskData.normalizeDistribution();
        display(log, img, maskData);
        return maskData;
      });
    }).collect(Collectors.toList());
    
    log.h3("Orthogonal Masks");
    List<Tensor> eigenMasks = refineMasks_pca(preblur(featureMasks, 4));
    eigenMasks.stream()
      .forEach(maskData -> {
        display(log, img, maskData);
      });
    
    log.h3("Clustered Masks");
    refineMasks_kmeans(eigenMasks).stream()
      .forEach(maskData -> {
        display(log, img, maskData);
      });
    
  }
  
  public abstract U getInstance();
  
  @Nonnull
  public abstract T[] getLayerTypes();
  
  /**
   * The type Vgg 19.
   */
  public static class VGG19 extends ImageSegmentation<CVPipe_VGG19.Layer, CVPipe_VGG19> {
    
    @Override
    public CVPipe_VGG19 getInstance() {
      return CVPipe_VGG19.INSTANCE;
    }
    
    @Override
    @Nonnull
    public CVPipe_VGG19.Layer[] getLayerTypes() {
      return CVPipe_VGG19.Layer.values();
    }
    
    
  }
}
