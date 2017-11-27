# SimpleConvolutionLayer
## SimpleConvolutionLayerTest
### Json Serialization
Code from [LayerTestBase.java:84](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L84) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f32.SimpleConvolutionLayer",
      "id": "0910987d-3688-428c-a892-e2c40000041e",
      "isFrozen": false,
      "name": "SimpleConvolutionLayer/0910987d-3688-428c-a892-e2c40000041e",
      "filter": {
        "dimensions": [
          3,
          3,
          1
        ],
        "data": [
          -1.152,
          0.108,
          1.412,
          1.712,
          0.116,
          0.844,
          0.56,
          -1.208,
          -0.764
        ]
      },
      "simple": false,
      "strideX": 1,
      "strideY": 1
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:121](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L121) executed in 0.00 seconds: 
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
    	[ [ 1.156 ], [ -0.384 ], [ 0.476 ] ],
    	[ [ -1.812 ], [ -1.904 ], [ 0.992 ] ],
    	[ [ 1.252 ], [ -1.14 ], [ -0.016 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -0.8161120414733887 ], [ -6.806735992431641 ], [ 1.1511517763137817 ] ],
    	[ [ 3.4743356704711914 ], [ 0.3078719973564148 ], [ 2.444431781768799 ] ],
    	[ [ -4.195663928985596 ], [ -0.46828824281692505 ], [ 3.667167901992798 ] ]
    ]
```



### Reference Implementation
Code from [LayerTestBase.java:132](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L132) executed in 0.02 seconds: 
```java
    System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(referenceLayer.getJson()));
    getEquivalencyTester().test(referenceLayer, layer, inputPrototype);
```
Logging: 
```
    {
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "0910987d-3688-428c-a892-e2c400000420",
      "isFrozen": false,
      "name": "ConvolutionLayer/0910987d-3688-428c-a892-e2c400000420",
      "filter": {
        "dimensions": [
          3,
          3,
          1
        ],
        "data": [
          -1.152,
          0.108,
          1.412,
          1.712,
          0.116,
          0.844,
          0.56,
          -1.208,
          -0.764
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
    	[ [ 1.156 ], [ -0.384 ], [ 0.476 ] ],
    	[ [ -1.812 ], [ -1.904 ], [ 0.992 ] ],
    	[ [ 1.252 ], [ -1.14 ], [ -0.016 ] ]
    ]
    Error: [
    	[ [ -4.147338827920066E-8 ], [ 7.568359272625003E-9 ], [ -2.236862182147803E-7 ] ],
    	[ [ -3.295288082405534E-7 ], [ -2.643585572670304E-9 ], [ -2.18231200665997E-7 ] ],
    	[ [ 7.10144041349281E-8 ], [ -2.4281692545535805E-7 ], [ -9.800720190966672E-8 ] ]
    ]
    Accuracy:
    absoluteTol: 1.3722e-07 +- 1.1164e-07 [2.6436e-09 - 3.2953e-07] (9#)
    relativeTol: 5.5618e-08 +- 7.7439e-08 [5.5595e-10 - 2.5926e-07] (9#)
    
```

### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.05 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.4976e-03 +- 1.5759e-03 [0.0000e+00 - 6.1882e-03] (162#)
    relativeTol: 1.6596e-01 +- 3.6745e-01 [9.8870e-07 - 1.0000e+00] (117#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.17 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 4.8245 +- 1.9356 [2.7472 - 10.4359]
    Learning performance: 6.0891 +- 2.5847 [3.1205 - 11.7155]
    
```

