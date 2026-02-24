# 2026-02-03T11:33:02.025538
import vitis

client = vitis.create_client()
client.set_workspace(path="MPPI_FLOAT_KINEMATIC")

vitis.dispose()

