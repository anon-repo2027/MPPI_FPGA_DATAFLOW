# 2026-02-16T09:32:17.592676
import vitis

client = vitis.create_client()
client.set_workspace(path="MPPI_FLOAT_KINEMATIC")

vitis.dispose()

