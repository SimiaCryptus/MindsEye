# ConvolutionLayer
## ConvolutionLayerTest
### Json Serialization
Code from [LayerTestBase.java:76](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L76) executed in 0.08 seconds: 
```java
    JsonObject json = layer.getJson();
    NNLayer echo = NNLayer.fromJson(json);
    assert (echo != null) : "Failed to deserialize";
    assert (layer != echo) : "Serialization did not copy";
    Assert.assertEquals("Serialization not equal", layer, echo);
    return new GsonBuilder().setPrettyPrinting().create().toJson(json);
```

Returns: 

```
    {
      "class": "com.simiacryptus.mindseye.layers.cudnn.f32.ConvolutionLayer",
      "id": "1ad638b7-7b0f-4c65-ad1d-b06200000001",
      "isFrozen": false,
      "name": "ConvolutionLayer/1ad638b7-7b0f-4c65-ad1d-b06200000001",
      "filter": {
        "dimensions": [
          3,
          3,
          4
        ],
        "data": [
          -0.5305982279979695,
          0.8914990257117348,
          -0.9541209563640727,
          -0.18428856233150626,
          -0.4074551125871795,
          -0.26613483772262847,
          0.6025260267042785,
          0.2388766215228728,
          0.18829629849775165,
          0.7815167734785269,
          -0.1478821908230421,
          0.6034940980515178,
          -0.20140353248287446,
          0.6403797502305699,
          -0.11490459860637303,
          -0.20373588704541623,
          -0.268081742826906,
          -0.10744696331078307,
          0.9553299166324147,
          0.7999474113198786,
          -0.43072628452684647,
          0.7047443893500027,
          0.05923480633010114,
          -0.13637075372095353,
          0.3583900768595023,
          0.5718203128030579,
          0.3044237895757904,
          -0.0053920186106151125,
          -0.6443101661623218,
          -0.7418680645718099,
          -0.3866996065028494,
          -0.5398233497478506,
          -0.8225502354070275,
          -0.25255724220401676,
          -0.4944588311297202,
          0.09416172769403097
        ]
      },
      "strideX": 1,
      "strideY": 1
    }
```



### Differential Validation
Code from [LayerTestBase.java:100](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L100) executed in 0.82 seconds: 
```java
    getDerivativeTester().test(layer, outputPrototype, inputPrototype);
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 5.4542e-04 +- 8.8031e-04 [0.0000e+00 - 6.3413e-03] (972#)
    relativeTol: 1.5543e-01 +- 3.5954e-01 [3.7319e-06 - 1.0000e+00] (463#)
    
```

### Performance
Code from [LayerTestBase.java:105](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L105) executed in 34.33 seconds: 
```java
    getPerformanceTester().test(layer, outputPrototype, inputPrototype);
```
Logging: 
```
    Forward performance: 1861.6056 +- 250.8385 [1598.3612 - 2903.5964]
    Backward performance: 1569.4114 +- 227.9160 [1421.2642 - 2876.0675]
    
```

### Reference Implementation
Code from [LayerTestBase.java:124](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L124) executed in 2.26 seconds: 
```java
    System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(referenceLayer.getJson()));
    getEquivalencyTester().test(referenceLayer, layer, outputPrototype, inputPrototype);
```
Logging: 
```
    {
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "1ad638b7-7b0f-4c65-ad1d-b06200002b49",
      "isFrozen": false,
      "name": "ConvolutionLayer/1ad638b7-7b0f-4c65-ad1d-b06200002b49",
      "filter": {
        "dimensions": [
          3,
          3,
          4
        ],
        "data": [
          -0.5305982279979695,
          0.8914990257117348,
          -0.9541209563640727,
          -0.18428856233150626,
          -0.4074551125871795,
          -0.26613483772262847,
          0.6025260267042785,
          0.2388766215228728,
          0.18829629849775165,
          0.7815167734785269,
          -0.1478821908230421,
          0.6034940980515178,
          -0.20140353248287446,
          0.6403797502305699,
          -0.11490459860637303,
          -0.20373588704541623,
          -0.268081742826906,
          -0.10744696331078307,
          0.9553299166324147,
          0.7999474113198786,
          -0.43072628452684647,
          0.7047443893500027,
          0.05923480633010114,
          -0.13637075372095353,
          0.3583900768595023,
          0.5718203128030579,
          0.3044237895757904,
          -0.0053920186106151125,
          -0.6443101661623218,
          -0.7418680645718099,
          -0.3866996065028494,
          -0.5398233497478506,
          -0.8225502354070275,
          -0.25255724220401676,
          -0.4944588311297202,
          0.09416172769403097
        ]
      },
      "skip": {
        "dimensions": [
          1,
          1
        ]
      },
      "simple": true
    }
    Reference Layer Accuracy:
    absoluteTol: 9.3582e-09 +- 1.8438e-08 [0.0000e+00 - 1.4082e-07] (972#)
    relativeTol: 3.6383e-08 +- 1.2935e-07 [5.0678e-10 - 2.4269e-06] (392#)
    
```

