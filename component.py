from kafka import KafkaClient, SimpleProducer, SimpleConsumer
from pyspark import SparkConf, SparkContext
from pyspark.mllib.classification import StreamingLogisticRegressionWithSGD, LogisticRegressionModel, \
    LogisticRegressionWithSGD
from pyspark.mllib.regression import LabeledPoint
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils


def send_kafka():
    kafka = KafkaClient('localhost:9092')
    producer = SimpleProducer(kafka)

    while True:
        producer.send_messages("data", b'data data data')
        producer.send_messages("weights", b'8.46,1.74,6.08,4.25,1.92')


def parse_point(row):
    """Parse input data. Last entry is the label"""
    if row[-1] == -1:
        label = 0
    elif row[-1] == 1:
        label = 1
    else:
        raise ValueError(row[-1])
    return LabeledPoint(label=label, features=row[:-1])


# online training model with pre-trained model to start
class MyStreamingLogisticRegressionWithSGD(StreamingLogisticRegressionWithSGD):
    """ Customized StreamingLogisticRegressionWithSGD
    with the ability to load pre-trained model
    """

    def __init__(self, trained_model, *args, **kwargs):
        super(MyStreamingLogisticRegressionWithSGD, self).__init__(*args, **kwargs)
        self.trained_model = trained_model
        self._model = LogisticRegressionModel(
            weights=self.trained_model.weights,
            intercept=self.trained_model.intercept,
            numFeatures=self.trained_model.numFeatures,
            numClasses=self.trained_model.numClasses,
        )

    def trainOn(self, dstream):
        """Train the model on the incoming dstream."""
        self._validate(dstream)

        def update(rdd):
            # LogisticRegressionWithSGD.train raises an error for an empty RDD.
            if not rdd.isEmpty():
                self._model = LogisticRegressionWithSGD.train(
                    rdd, self.numIterations, self.stepSize,
                    self.miniBatchFraction, self._model.weights,
                    regParam=self.regParam, convergenceTol=self.convergenceTol)

        dstream.foreachRDD(update)


def get_model(weight, pretrained=True):
    """ Initiate a streaming model."""
    if pretrained:
        trained_model = _load_pre_trained_model()
        model = MyStreamingLogisticRegressionWithSGD(trained_model=trained_model)
    else:
        model = StreamingLogisticRegressionWithSGD()
        model.setInitialWeights(weight)
    return model


def _load_pre_trained_model():
    """ load trained LogisticRegressionModel model"""
    trained_model = LogisticRegressionModel.load(sc, "model/SGD")
    return trained_model


if __name__ == '__main__':

    # Streaming context
    conf = SparkConf().setMaster("local[2]")
    sc = SparkContext.getOrCreate(conf=conf)
    ssc = StreamingContext(sc, 1)

    # Kafka consumer for Component weight
    kafka = KafkaClient('localhost:9092')
    consumer = SimpleConsumer(kafka, topic="weights", group="consumer", auto_offset_reset='latest')
    weights = consumer.get_message()[1]
    weight = weights.value.decode('utf-8').split(',')

    # Get model
    model = get_model(weight, pretrained=False)

    # load data from Kafka
    directKafkaStream = KafkaUtils.createDirectStream(
        ssc,
        ["data"],
        {"metadata.broker.list": "localhost:9092"}
    )

    # parse
    test_data = directKafkaStream.map(
        lambda line: line[1].split(',')
    ).map(
        lambda row: [int(x) for x in row]
    ).map(
        parse_point
    )

    # Predict and Train
    test_data.map(
        lambda row: [
            # predict with pre-trained model
            # model_with_pretrain._model.predict(row.features),

            model._model.predict(row.features),  # predict with NO pre-trained model
            row.label]  # predict the result
    )

    # online train model
    # model_with_pretrain.trainOn(test_data)
    model.trainOn(test_data)  # online train model

    # Start the streaming context
    ssc.start()
    ssc.awaitTermination()
