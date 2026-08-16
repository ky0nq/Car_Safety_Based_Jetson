from security_policy import UnknownOnlyCapturePolicy


def test_policy() -> None:
    policy = UnknownOnlyCapturePolicy(10.0, 20.0, 30.0)

    s = policy.update(False, False, True, now=0.0)
    assert not s.should_capture
    assert not s.should_alarm
    assert s.unknown_presence_elapsed == 0.0

    # 10 s -> capture once.
    s = policy.update(False, False, True, now=10.0)
    assert s.should_capture
    assert not s.should_alarm

    # 20 s -> only capture-cycle timer resets; continuous presence remains 20 s.
    s = policy.update(False, False, True, now=20.0)
    assert s.unknown_elapsed == 0.0
    assert s.unknown_presence_elapsed == 20.0
    assert not s.should_alarm

    # 30 s continuous UNKNOWN -> capture for second cycle + alarm once.
    s = policy.update(False, False, True, now=30.0)
    assert s.should_capture
    assert s.should_alarm
    assert s.alarm_triggered
    assert s.state == "ALARM"

    # Alarm does not retrigger continuously.
    s = policy.update(False, False, True, now=35.0)
    assert not s.should_alarm
    assert s.alarm_triggered

    # Registered person resets continuous presence and alarm state.
    s = policy.update(True, False, True, now=36.0)
    assert s.state == "REGISTERED_PRESENT"
    assert s.unknown_presence_elapsed == 0.0
    assert not s.alarm_triggered

    # New UNKNOWN episode must wait a fresh 30 s for alarm.
    policy.update(False, False, True, now=40.0)
    s = policy.update(False, False, True, now=69.9)
    assert not s.should_alarm
    s = policy.update(False, False, True, now=70.0)
    assert s.should_alarm


if __name__ == "__main__":
    test_policy()
    print("policy test passed")
