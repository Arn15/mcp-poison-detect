from mcptp.eval.calibration import RegistryCalibrator, logit, worst_group_threshold


def test_registry_relative_cancels_server_offset_and_excludes_same_name():
    ref = [("A", "t1", 0.01), ("A", "t2", 0.02), ("B", "u1", 0.5)]
    cal = RegistryCalibrator(ref, stat="max")
    # same raw score is far more suspicious on quiet server A than on noisy server B
    assert cal.transform("A", "new", 0.5) > 0 > cal.transform("B", "new", 0.3)
    # a candidate is never compared with a registered tool of the same name (paraphrase of itself)
    assert abs(cal.transform("A", "t2", 0.02) - (logit(0.02) - logit(0.01))) < 1e-9
    assert cal.transform("C", "x", 0.9) is None          # no registry -> no calibration
    assert RegistryCalibrator(ref, stat="median").transform("A", "new", 0.5) > cal.transform("A", "new", 0.5)


def test_worst_group_threshold_is_max_of_groups():
    t = worst_group_threshold({"a": [0.1] * 99 + [0.9], "b": [0.2] * 100}, 0.01)
    assert t > logit(0.2)
