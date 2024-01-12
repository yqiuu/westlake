import westlake


def test_save_config(tmp_path):
    """This only tests if the config file is saved but does not check if the
    config file is correctly.
    """
    fname = tmp_path/"config.yml"
    westlake.save_config_template(fname)
    fname.exists()