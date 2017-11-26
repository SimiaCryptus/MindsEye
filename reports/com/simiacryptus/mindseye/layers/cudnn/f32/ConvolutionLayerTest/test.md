# ConvolutionLayer
## ConvolutionLayerTest
### Json Serialization
Code from [LayerTestBase.java:75](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L75) executed in 0.00 seconds: 
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
      "id": "055e4cd6-0193-4699-9154-1c1700000028",
      "isFrozen": false,
      "name": "ConvolutionLayer/055e4cd6-0193-4699-9154-1c1700000028",
      "filter": {
        "dimensions": [
          3,
          3,
          4
        ],
        "data": [
          1.176,
          1.772,
          0.224,
          -1.692,
          -0.848,
          0.492,
          1.52,
          -0.46,
          -1.18,
          -1.584,
          0.544,
          -1.856,
          1.408,
          0.368,
          1.5,
          -0.676,
          -1.976,
          1.14,
          1.7,
          -0.72,
          1.98,
          -0.56,
          0.988,
          0.612,
          0.58,
          -1.064,
          1.944,
          -1.512,
          0.932,
          -1.728,
          -1.416,
          -0.272,
          -0.884,
          -0.956,
          -0.4,
          0.328
        ]
      },
      "strideX": 1,
      "strideY": 1
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:112](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L112) executed in 0.01 seconds: 
```java
    SimpleEval eval = SimpleEval.run(layer, inputPrototype);
    return String.format("--------------------\nInput: \n[%s]\n--------------------\nOutput: \n%s",
      Arrays.stream(inputPrototype).map(t->t.prettyPrint()).reduce((a,b)->a+",\n"+b).get(),
      eval.getOutput().prettyPrint());
```

Returns: 

```
    --------------------
    Input: 
    [[
    	[ [ -1.82, -1.12 ], [ -0.216, 0.272 ], [ 0.176, 0.528 ] ],
    	[ [ 0.396, 1.512 ], [ -0.836, -0.724 ], [ 1.092, 1.732 ] ],
    	[ [ -1.524, 1.076 ], [ 1.532, 1.632 ], [ -0.172, 1.612 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -3.8724799156188965, 0.6063680648803711 ], [ 9.939918518066406, -1.7348957061767578 ], [ -4.3258562088012695, 0.5814723968505859 ] ],
    	[ [ 5.6593122482299805, -11.76780891418457 ], [ -2.6542880535125732, -5.852574348449707 ], [ 5.796975612640381, -3.608079433441162 ] ],
    	[ [ 3.3944954872131348, 3.561008930206299 ], [ 3.695647716522217, -0.576545238494873 ], [ 0.4736166000366211, -5.265392303466797 ] ]
    ]
```



### Reference Implementation
Code from [LayerTestBase.java:123](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L123) executed in 0.02 seconds: 
```java
    System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(referenceLayer.getJson()));
    getEquivalencyTester().test(referenceLayer, layer, inputPrototype);
```
Logging: 
```
    {
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "055e4cd6-0193-4699-9154-1c1700000030",
      "isFrozen": false,
      "name": "ConvolutionLayer/055e4cd6-0193-4699-9154-1c1700000030",
      "filter": {
        "dimensions": [
          3,
          3,
          4
        ],
        "data": [
          1.176,
          1.772,
          0.224,
          -1.692,
          -0.848,
          0.492,
          1.52,
          -0.46,
          -1.18,
          -1.584,
          0.544,
          -1.856,
          1.408,
          0.368,
          1.5,
          -0.676,
          -1.976,
          1.14,
          1.7,
          -0.72,
          1.98,
          -0.56,
          0.988,
          0.612,
          0.58,
          -1.064,
          1.944,
          -1.512,
          0.932,
          -1.728,
          -1.416,
          -0.272,
          -0.884,
          -0.956,
          -0.4,
          0.328
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
    Inputs: [
    	[ [ -1.82, -1.12 ], [ -0.216, 0.272 ], [ 0.176, 0.528 ] ],
    	[ [ 0.396, 1.512 ], [ -0.836, -0.724 ], [ 1.092, 1.732 ] ],
    	[ [ -1.524, 1.076 ], [ 1.532, 1.632 ], [ -0.172, 1.612 ] ]
    ]
    Error: [
    	[ [ 8.438110343789162E-8, 6.488037096463728E-8 ], [ -1.4819335927285238E-6, 2.9382324151505657E-7 ], [ -2.0880126960776124E-7, 3.968505857265825E-7 ] ],
    	[ [ 2.482299805706134E-7, -9.141845715987529E-7 ], [ -5.3512573039427025E-8, 1.6515502920810832E-6 ], [ -3.8735961993552337E-7, 5.665588376224662E-7 ] ],
    	[ [ -5.127868658583168E-7, 9.302062986549231E-7 ], [ -2.834777839133551E-7, -1.2384948734345258E-6 ], [ 6.000366207237207E-7, -3.034667965806648E-7 ] ]
    ]
    Accuracy:
    absoluteTol: 5.6781e-07 +- 4.7052e-07 [5.3513e-08 - 1.6516e-06] (18#)
    relativeTol: 1.6076e-07 +- 2.6633e-07 [1.0080e-08 - 1.0741e-06] (18#)
    
```

### Differential Validation
Code from [LayerTestBase.java:130](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L130) executed in 0.25 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.9144e-04 +- 2.9580e-04 [0.0000e+00 - 2.2568e-03] (972#)
    relativeTol: 1.5543e-01 +- 3.6196e-01 [3.3181e-06 - 1.0000e+00] (464#)
    
```

### Performance
Code from [LayerTestBase.java:135](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L135) executed in 24.74 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 8.8036 +- 5.8964 [5.6426 - 275.7052]
    Learning performance: 7.7606 +- 3.1705 [5.1781 - 231.0177]
    
```

