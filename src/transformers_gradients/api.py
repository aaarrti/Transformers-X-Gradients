class text_classification(object):
    @staticmethod
    def gradient_norm(*args, **kwargs):
        from transformers_gradients.tasks.text_classification import gradient_norm

        return gradient_norm(*args, **kwargs)

    @staticmethod
    def gradient_x_input(*args, **kwargs):
        from transformers_gradients.tasks.text_classification import gradient_x_input

        return gradient_x_input(*args, **kwargs)

    @staticmethod
    def integrated_gradients(*args, **kwargs):
        from transformers_gradients.tasks.text_classification import (
            integrated_gradients,
        )

        return integrated_gradients(*args, **kwargs)

    @staticmethod
    def smooth_grad(*args, **kwargs):
        from transformers_gradients.tasks.text_classification import smooth_grad

        return smooth_grad(*args, **kwargs)

    @staticmethod
    def noise_grad(*args, **kwargs):
        from transformers_gradients.tasks.text_classification import smooth_grad

        return smooth_grad(*args, **kwargs)

    @staticmethod
    def noise_grad_plus_plus(*args, **kwargs):
        from transformers_gradients.tasks.text_classification import (
            noise_grad_plus_plus,
        )

        return noise_grad_plus_plus(*args, **kwargs)

    @staticmethod
    def lime(*args, **kwargs):
        from transformers_gradients.tasks.text_classification import lime

        return lime(*args, **kwargs)
