# ConvolutionLayer
## ConvolutionLayerTest
### Json Serialization
Code from [LayerTestBase.java:76](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L76) executed in 0.10 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f64.ConvolutionLayer",
      "id": "b969dd9a-e5f5-40b4-b562-cd8600000001",
      "isFrozen": false,
      "name": "ConvolutionLayer/b969dd9a-e5f5-40b4-b562-cd8600000001",
      "filter": {
        "dimensions": [
          3,
          3,
          4
        ],
        "data": [
          0.8347113421805339,
          0.4640440311740892,
          -0.5844500098916485,
          -0.22486813140329276,
          -0.7184358863937916,
          -0.22428506293358064,
          -0.5001863141144582,
          0.8111997971036773,
          -0.9100777250685907,
          -0.4661891966398293,
          -0.6903857302320264,
          -0.9195988950652367,
          0.940648729975079,
          -0.7262856330233398,
          0.92064115077565,
          -0.3206085012340816,
          0.20497045258118263,
          0.5315363971996447,
          -0.2715228848273741,
          -0.41882036875516393,
          -0.5859544888308026,
          -0.4079165824897437,
          -0.7563834973921431,
          -0.29818414598322196,
          0.83891023326256,
          0.20217804919449023,
          -0.6721542254552366,
          -0.16118366415620522,
          0.3344585420811912,
          0.8002361008425178,
          -0.5106952254504671,
          0.04863149369383657,
          0.05852970698071114,
          -0.9846491137912532,
          0.2899337326907041,
          0.4829182912369234
        ]
      },
      "strideX": 1,
      "strideY": 1
    }
```



### Differential Validation
Code from [LayerTestBase.java:100](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L100) executed in 0.84 seconds: 
```java
    getDerivativeTester().test(layer, outputPrototype, inputPrototype);
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 5.6524e-11 +- 1.1532e-10 [0.0000e+00 - 8.2794e-10] (972#)
    relativeTol: 2.3630e-10 +- 6.1952e-10 [6.5288e-13 - 6.2336e-09] (392#)
    
```

### Performance
Code from [LayerTestBase.java:105](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L105) executed in 30.82 seconds: 
```java
    getPerformanceTester().test(layer, outputPrototype, inputPrototype);
```
Logging: 
```
    Forward performance: 1631.4664 +- 261.0600 [1444.1593 - 2795.0823]
    Backward performance: 1447.9109 +- 192.0346 [1213.2670 - 2423.5959]
    
```

### Reference Implementation
Code from [LayerTestBase.java:124](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L124) executed in 2.29 seconds: 
```java
    System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(referenceLayer.getJson()));
    getEquivalencyTester().test(referenceLayer, layer, outputPrototype, inputPrototype);
```
Logging: 
```
    {
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "b969dd9a-e5f5-40b4-b562-cd8600002b49",
      "isFrozen": false,
      "name": "ConvolutionLayer/b969dd9a-e5f5-40b4-b562-cd8600002b49",
      "filter": {
        "dimensions": [
          3,
          3,
          4
        ],
        "data": [
          0.8347113421805339,
          0.4640440311740892,
          -0.5844500098916485,
          -0.22486813140329276,
          -0.7184358863937916,
          -0.22428506293358064,
          -0.5001863141144582,
          0.8111997971036773,
          -0.9100777250685907,
          -0.4661891966398293,
          -0.6903857302320264,
          -0.9195988950652367,
          0.940648729975079,
          -0.7262856330233398,
          0.92064115077565,
          -0.3206085012340816,
          0.20497045258118263,
          0.5315363971996447,
          -0.2715228848273741,
          -0.41882036875516393,
          -0.5859544888308026,
          -0.4079165824897437,
          -0.7563834973921431,
          -0.29818414598322196,
          0.83891023326256,
          0.20217804919449023,
          -0.6721542254552366,
          -0.16118366415620522,
          0.3344585420811912,
          0.8002361008425178,
          -0.5106952254504671,
          0.04863149369383657,
          0.05852970698071114,
          -0.9846491137912532,
          0.2899337326907041,
          0.4829182912369234
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
    absoluteTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (972#)
    relativeTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (392#)
    
```

