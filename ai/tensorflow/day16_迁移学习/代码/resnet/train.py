from tensorflow import keras
from model import resnet34, resnet50



train_dir = r'D:\AIoT-深度学习视频版\深度学习二期\day12_ALexNet和VGG\代码\training\training'
valid_dir = r'D:\AIoT-深度学习视频版\深度学习二期\day12_ALexNet和VGG\代码\validation\validation'
label_file = r'D:\AIoT-深度学习视频版\深度学习二期\day12_ALexNet和VGG\代码\monkey_labels.txt'


def main():
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
            rescale=1.0 / 255.0,
            # 以下都是图片数据增强操作.
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            samplewise_std_normalization=True,
            fill_mode='nearest'
    )

    # 从目录中读取图片
    height = 224
    width = 224
    channels = 3
    batch_size = 64
    num_classes = 10

    # 会自动把目录名作为label名.
    train_generator = train_datagen.flow_from_directory(train_dir, target_size=(
    height, width),
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        seed=7,
                                                        class_mode='categorical')

    valid_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255.0, samplewise_std_normalization=True)

    valid_generator = valid_datagen.flow_from_directory(valid_dir, target_size=(
    height, width),
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        class_mode='categorical')
    model34 = resnet34(num_classes=10)
    model34.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    train_num = train_generator.samples
    valid_num = valid_generator.samples

    history = model34.fit(train_generator,
                        steps_per_epoch=train_num // batch_size,
                        epochs=30,
                        validation_data=valid_generator,
                        validation_steps=valid_num // batch_size)



if __name__ == '__main__':
    main()
