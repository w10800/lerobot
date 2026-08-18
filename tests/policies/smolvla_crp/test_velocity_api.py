from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching


class DummyVelocityModel:
    def __init__(self):
        self.called_with = None

    def predict_velocity(self, **kwargs):
        self.called_with = kwargs
        return "velocity"


def test_denoise_step_delegates_to_velocity_api_without_changing_arguments():
    dummy = DummyVelocityModel()
    result = VLAFlowMatching.denoise_step(
        dummy,
        prefix_pad_masks="prefix",
        past_key_values="cache",
        x_t="x",
        timestep="t",
        target_time="s",
    )
    assert result == "velocity"
    assert dummy.called_with == {
        "prefix_pad_masks": "prefix",
        "past_key_values": "cache",
        "x_t": "x",
        "timestep": "t",
        "target_time": "s",
    }
