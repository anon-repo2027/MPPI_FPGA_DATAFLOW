# 2026-01-22T13:50:45.509754
import vitis

client = vitis.create_client()
client.set_workspace(path="MPPI_FLOAT_KINEMATIC")

comp = client.get_component(name="MPPI_Float")
comp.run(operation="SYNTHESIS")

vitis.dispose()

